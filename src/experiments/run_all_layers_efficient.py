import os
import sys
import importlib.util
import gc
import glob
import time
import pickle
import gzip
import yaml
import numpy as np
import torch
from tqdm import tqdm
import traceback

# 1. Safely hide broken torchaudio C++ extension from transformers/evaluate without polluting sys.modules
_orig_find_spec = importlib.util.find_spec
def _safe_find_spec(name, *args, **kwargs):
    if name == 'torchaudio' or (isinstance(name, str) and name.startswith('torchaudio.')):
        return None
    return _orig_find_spec(name, *args, **kwargs)
importlib.util.find_spec = _safe_find_spec

# 1.1 Compatibility Shim for Phi-3 / DeepSeek MoE with modern transformers DynamicCache ('seen_tokens', 'get_max_length' errors)
try:
    from transformers.cache_utils import DynamicCache, Cache
    if not hasattr(DynamicCache, "seen_tokens"):
        def _get_seen_tokens(self, layer_idx=0):
            try:
                return self.get_seq_length(layer_idx)
            except Exception:
                try:
                    return self.get_seq_length()
                except Exception:
                    return len(self.key_cache[0][0]) if (hasattr(self, 'key_cache') and self.key_cache) else 0
        DynamicCache.seen_tokens = property(_get_seen_tokens)
        DynamicCache._seen_tokens = property(_get_seen_tokens)

    _orig_cache_getitem = getattr(DynamicCache, "__getitem__", None)
    def _safe_cache_getitem(self, item):
        if item in ("seen_tokens", "_seen_tokens"):
            try:
                return self.get_seq_length()
            except Exception:
                return 0
        if _orig_cache_getitem is not None:
            return _orig_cache_getitem(self, item)
        if isinstance(item, int):
            return (self.key_cache[item], self.value_cache[item])
        raise KeyError(item)
    DynamicCache.__getitem__ = _safe_cache_getitem

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, seq_len=None, layer_idx=0):
            try:
                return self.get_seq_length(layer_idx)
            except Exception:
                return seq_len if seq_len is not None else 0
        DynamicCache.get_usable_length = _get_usable_length

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            return getattr(self, "max_cache_length", None)
        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "get_max_cache_shape"):
        def _get_max_cache_shape(self):
            return getattr(self, "max_cache_length", None)
        DynamicCache.get_max_cache_shape = _get_max_cache_shape

    for attr, fn in [
        ("get_max_length", lambda self: getattr(self, "max_cache_length", None)),
        ("get_max_cache_shape", lambda self: getattr(self, "max_cache_length", None)),
        ("get_usable_length", lambda self, seq_len=None, layer_idx=0: self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else seq_len)
    ]:
        if 'Cache' in locals() and not hasattr(Cache, attr):
            setattr(Cache, attr, fn)
except Exception:
    pass

# 2. Dynamic path resolution to root, src, and experiments
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
experiments_dir = os.path.abspath(os.path.join(src_dir, 'experiments'))
for p in [script_dir, src_dir, experiments_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from all_layers_args import AllLayersArgs

# 3. Authenticate HF Hub via args.hf_token
args_pre = AllLayersArgs()
if getattr(args_pre, 'hf_token', None):
    os.environ["HF_TOKEN"] = args_pre.hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = args_pre.hf_token

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
from utils import seed_everything, generate_text, compute_likelihood
from disagree import calculate_all_disagreement_metrics
from token_attribution import compute_token_score_ranking, rank_tensor
from layer_ensemble import get_model_layer_count


def evaluate_all_layers_for_sample(args, model, tokenizer, deberta_model, deberta_tokenizer,
                                  deberta_embeddings, device_llm, device_deberta,
                                  input_ids, prompt, question, initial_generation, initial_likelihood):
    """
    Efficient single-pass all-layers hidden state extraction and multi-depth metric computation.
    Does NOT require continuing multi-token autoregressive decoding after substitution.
    """
    initial_generation_text = initial_generation['generation_text'][0]
    initial_generation_ids = initial_generation['generation_ids'][0]
    
    if len(initial_likelihood["generation_logits"]) == 0:
        return []
        
    generation_logits = initial_likelihood["generation_logits"][0].to(dtype=torch.float32)
    generation_logits = torch.nn.functional.softmax(generation_logits, dim=-1) + 1e-9
    
    if initial_generation_ids.shape[0] == 0 or len(initial_generation_text.split()) == 0:
        return []

    # Rank candidate substitution tokens via DeBERTa
    single_word = len(initial_generation_text.split()) == 1
    if single_word:
        token_info = {}
        if args.token_prob_threshold is None:
            other_voc_indices = torch.tensor(range(len(generation_logits[0])))
        else:
            other_voc_indices = torch.where(generation_logits[0] > args.token_prob_threshold)[0]
        for new_token_idx in other_voc_indices:
            if new_token_idx != initial_generation_ids[0] and new_token_idx not in args.invalid_ids:
                importance_score = generation_logits[0][new_token_idx]
                token_info[(0, 0, new_token_idx.item())] = (0, 0, importance_score.item())
        sorted_indices = torch.argsort(rank_tensor([v[2] for v in token_info.values()], descending=True), descending=False)
    else:
        additional_generated_text = []
        sorted_indices, token_info = compute_token_score_ranking(
            deberta_model, deberta_tokenizer, device_deberta,
            question, initial_generation_ids, initial_generation_text,
            additional_generated_text, generation_logits,
            deberta_embeddings, tokenizer, args
        )
        if not isinstance(sorted_indices, torch.Tensor):
            return []

    token_info_list = list(token_info.keys())
    total_layers = get_model_layer_count(model)
    lm_head = model.get_output_embeddings()
    k_75 = max(1, int(round(args.depth_ratio * total_layers)))

    sample_epistemic_records = []
    num_evals = 0

    with torch.no_grad():
        for s in sorted_indices:
            initial_gen_word_idx, initial_gen_token_idx, new_token_idx = token_info_list[s]
            if initial_gen_token_idx > 0:
                new_input_ids = initial_generation_ids[:initial_gen_token_idx]
                token_to_replace_id = initial_generation_ids[initial_gen_token_idx]
                all_input_ids = torch.hstack([input_ids.to(device_llm), new_input_ids.unsqueeze(0).to(device_llm)])
            else:
                initial_gen_token_idx = 0
                all_input_ids = input_ids.to(device_llm)
                token_to_replace_id = initial_generation_ids[0]

            if new_token_idx in args.invalid_ids or new_token_idx == token_to_replace_id.item():
                continue

            # Single forward pass to extract all hidden states
            model_outputs = model(all_input_ids, return_dict=True, output_hidden_states=True)
            final_logits = model_outputs.logits[:, -1, :].float()

            # Stack all layers (retaining 100% of layer depth for EU) - 1 single batched kernel launch for all layers
            target_dev = getattr(lm_head, 'weight', None).device if (hasattr(lm_head, 'weight') and hasattr(lm_head.weight, 'device')) else final_logits.device
            hs_stack = torch.stack([model_outputs.hidden_states[l][:, -1, :].to(target_dev) for l in range(total_layers)], dim=0)
            all_logits = lm_head(hs_stack) # 1 single batched kernel launch for all layers
            all_premature_logits = {l: all_logits[l].to(final_logits.device) for l in range(total_layers)}

            step_metrics = {}

            # 1. Full layer sweep: compute for all depths k in 1 ... total_layers-1
            for k in range(1, total_layers):
                candidate_premature_layers = list(range(total_layers - k, total_layers))
                cand_logits = {l: all_premature_logits[l] for l in candidate_premature_layers}
                m_res = calculate_all_disagreement_metrics(
                    final_logits, cand_logits,
                    relative_top=args.relative_top,
                    min_tokens_to_keep=args.min_tokens_to_keep,
                    skip_top_neighbors=args.skip_top_neighbors
                )
                
                step_metrics[f'entropy_mixture_top_{k}'] = m_res.get('entropy_mixture', 0.0)
                step_metrics[f'mutual_info_top_{k}'] = m_res.get('mutual_info', 0.0)
                step_metrics[f'gated_mutual_info_top_{k}'] = m_res.get('gated_mutual_info', 0.0)
                
                if k > 2:
                    step_metrics[f'mutual_info_top_{k}_skip2'] = m_res.get('mutual_info_skip2', m_res.get('mutual_info', 0.0))
                    step_metrics[f'gated_mutual_info_top_{k}_skip2'] = m_res.get('gated_mutual_info_skip2', m_res.get('gated_mutual_info', 0.0))

            # 2. Depth normalized (75% depth) metrics
            cand_logits_75 = {l: all_premature_logits[l] for l in range(total_layers - k_75, total_layers)}
            m_res_75 = calculate_all_disagreement_metrics(
                final_logits, cand_logits_75,
                relative_top=args.relative_top,
                min_tokens_to_keep=args.min_tokens_to_keep,
                skip_top_neighbors=args.skip_top_neighbors
            )
            step_metrics['entropy_mixture_depth_75'] = m_res_75.get('entropy_mixture', 0.0)
            step_metrics['mutual_info_depth_75'] = m_res_75.get('mutual_info', 0.0)
            step_metrics['gated_mutual_info_depth_75'] = m_res_75.get('gated_mutual_info', 0.0)
            step_metrics['mutual_info_depth_75_skip2'] = m_res_75.get('mutual_info_skip2', m_res_75.get('mutual_info', 0.0))
            step_metrics['gated_mutual_info_depth_75_skip2'] = m_res_75.get('gated_mutual_info_skip2', m_res_75.get('gated_mutual_info', 0.0))

            # 3. Standard baseline keys (untouched for backward compatibility)
            k_a_default = min(2, total_layers - 1)
            k_e_default = min(16, total_layers - 1)
            step_metrics['entropy_final'] = m_res_75.get('entropy_final', 0.0)
            step_metrics['entropy_mixture'] = step_metrics.get(f'entropy_mixture_top_{k_a_default}', m_res_75.get('entropy_mixture', 0.0))
            step_metrics['mutual_info'] = step_metrics.get(f'mutual_info_top_{k_e_default}', m_res_75.get('mutual_info', 0.0))
            step_metrics['gated_mutual_info'] = step_metrics.get(f'gated_mutual_info_top_{k_e_default}', m_res_75.get('gated_mutual_info', 0.0))
            step_metrics['jsd'] = m_res_75.get('jsd', 0.0)
            step_metrics['kl'] = m_res_75.get('kl', 0.0)
            step_metrics['epkl'] = m_res_75.get('epkl', 0.0)
            step_metrics['cv'] = m_res_75.get('cv', 0.0)
            step_metrics['tvd'] = m_res_75.get('tvd', 0.0)
            step_metrics['bhattacharyya'] = m_res_75.get('bhattacharyya', 0.0)

            sample_epistemic_records.append(step_metrics)
            num_evals += 1
            if num_evals >= args.num_total_generations:
                break

    return sample_epistemic_records


def compute_batched_bleurt(bleurt_metric, sample_pairs_list, batch_size=64):
    """
    Performs vectorized GPU batched computation of BLEURT scores.
    sample_pairs_list: list of tuples (sample_idx, gen_text, ref_answers_list)
    Returns: dict mapping sample_idx -> max_bleurt_score
    """
    if bleurt_metric is None or not sample_pairs_list:
        return {}

    flat_preds = []
    flat_refs = []
    sample_mapping = []  # list of (sample_idx, start_pos, end_pos)

    for s_idx, gen_text, refs in sample_pairs_list:
        start_p = len(flat_preds)
        for r in refs:
            flat_preds.append(gen_text)
            flat_refs.append(r)
        end_p = len(flat_preds)
        sample_mapping.append((s_idx, start_p, end_p))

    all_scores = []
    total_pairs = len(flat_preds)
    print(f"[INFO] Computing vectorized BLEURT across {total_pairs} pair evaluations (batch_size={batch_size})...")

    for b_start in tqdm(range(0, total_pairs, batch_size), desc="BLEURT Batch Vector Inference"):
        b_end = min(b_start + batch_size, total_pairs)
        chunk_preds = flat_preds[b_start:b_end]
        chunk_refs = flat_refs[b_start:b_end]
        try:
            res = bleurt_metric.compute(predictions=chunk_preds, references=chunk_refs)
            all_scores.extend(res['scores'])
        except Exception as e:
            print(f"[ERROR] BLEURT batch [{b_start}:{b_end}] failed: {e}")
            all_scores.extend([0.0] * (b_end - b_start))

    # Map scores back to each sample as max score across reference answers
    results_map = {}
    for s_idx, start_p, end_p in sample_mapping:
        if end_p > start_p and end_p <= len(all_scores):
            results_map[s_idx] = max(all_scores[start_p:end_p])
        else:
            results_map[s_idx] = 0.0

    return results_map


def run_pipeline(args=None):
    if args is None:
        args = AllLayersArgs()

    run_id = args.run_id
    canonical_result_dir = os.path.join(experiments_dir, 'result')
    base_path = os.path.join(canonical_result_dir, run_id)
    os.makedirs(base_path, exist_ok=True)
    args.args_to_yaml(base_path)

    # Prepare entropies3 directory for final analysis
    base_path_entropies = os.path.join(canonical_result_dir, getattr(args, 'entropies_dir', 'entropies3'))
    os.makedirs(base_path_entropies, exist_ok=True)

    # Derive clean tags for entropies3 file naming
    dataset_lower = args.dataset.lower()
    if 'truth' in dataset_lower:
        data_tag = 'truth'
    elif 'open' in dataset_lower:
        data_tag = 'open'
    elif 'science' in dataset_lower:
        data_tag = 'scienceqa'
    else:
        data_tag = dataset_lower

    model_clean = args.llm_model.split('/')[-1].lower().replace('-instruct', '').replace('-chat', '').replace('_', '-')
    modl_tag = 'qwen-1.5b' if 'qwen2.5-1.5b' in model_clean else model_clean
    quant_suffix = f"_{getattr(args, 'quantization', '')}" if getattr(args, 'quantization', None) else ""
    modl_tag_with_quant = f"{modl_tag}{quant_suffix}"

    print("=" * 80)
    print(f"RUNNING ALL-LAYERS EPISTEMIC UNCERTAINTY PIPELINE: {run_id}")
    print(f"Model: {args.llm_model} | Output Directory: {base_path}")
    print(f"Target Entropies3 Directory: {base_path_entropies}")
    print(f"BLEURT Batch Size: {getattr(args, 'bleurt_batch_size', 64)}")
    print("=" * 80)

    seed_everything(args.seed_value)
    device_llm = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_deberta = device_llm

    # =========================================================================
    # PHASE 1: GENERATION & ALL-LAYERS IN-SITU EXTRACTION (PyTorch GPU)
    # =========================================================================
    print(f"\n[PHASE 1] Initializing PyTorch models on {device_llm}...")
    is_phi = "phi-3" in args.llm_model.lower() or "phi3" in args.llm_model.lower()
    trust_remote = False if is_phi else True
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, token=args.hf_token, trust_remote_code=trust_remote)
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{args.deberta_model}", token=args.hf_token)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{args.deberta_model}",
        token=args.hf_token
    ).to(device_deberta)
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()

    # Setup quantization config if requested
    quantization_config = None
    quant_mode = getattr(args, 'quantization', None)
    if quant_mode in ['4bit', '8bit'] and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        if quant_mode == '4bit':
            print("Configuring 4-bit (NF4) quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quant_mode == '8bit':
            print("Configuring 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

    attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        token=args.hf_token,
        trust_remote_code=trust_remote,
        attn_implementation=attn_impl,
        device_map='auto' if torch.cuda.is_available() else None
    )
    if device_llm != 'cpu' and not hasattr(llm_model, 'hf_device_map') and quantization_config is None:
        llm_model = llm_model.to(device_llm)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else '[PAD]'
        if hasattr(llm_model, 'config'):
            llm_model.config.pad_token_id = tokenizer.pad_token_id

    # EOS token resolution
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.extend(tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id])
    for s_tok in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "."]:
        tid = tokenizer.convert_tokens_to_ids(s_tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    args.eos_token_ids = list(set(eos_ids))

    # Load dataset dynamically based on args.dataset
    dataset_name = str(args.dataset).lower()
    print(f"[INFO] Loading dataset '{args.dataset}'...")
    if 'science' in dataset_name:
        try:
            dataset = load_dataset('tasksource/ScienceQA_text_only')
        except Exception:
            dataset = load_dataset('allenai/sciq')
        split_name = 'test' if 'test' in dataset else ('validation' if 'validation' in dataset else 'train')
        val_set = dataset[split_name]
    elif 'open' in dataset_name:
        try:
            dataset = load_dataset('openbookqa', 'main')
        except Exception:
            try:
                dataset = load_dataset('openbookqa')
            except Exception:
                dataset = load_dataset('allenai/openbookqa', 'additional')
        split_name = 'test' if 'test' in dataset else ('validation' if 'validation' in dataset else 'train')
        val_set = dataset[split_name]
    elif 'coqa' in dataset_name:
        try:
            dataset = load_dataset('stanfordnlp/coqa')
        except Exception:
            dataset = load_dataset('coqa')
        val_set = dataset['validation']
    else:  # truthful_qa / truthfulqa (default)
        try:
            dataset = load_dataset('truthfulqa/truthful_qa', 'generation')
        except Exception:
            dataset = load_dataset('truthful_qa', 'generation')
        val_set = dataset['validation']

    start_idx = getattr(args, 'start_idx', 0)
    total_len = len(val_set)
    if args.num_samples == -1 or start_idx + args.num_samples > total_len:
        num_samples = total_len - start_idx
    else:
        num_samples = args.num_samples
    end_idx = start_idx + num_samples

    print(f"[INFO] Running generation & all-layers extraction on {num_samples} samples (indices {start_idx} to {end_idx - 1} of {total_len})...")

    for idx in tqdm(range(start_idx, end_idx), desc="Phase 1: Generation & Layer Extraction"):
        sample = val_set[idx]
        k_id = idx
        out_file = os.path.join(base_path, f'results_dict_{k_id}.pkl.gz' if args.compress_pkl else f'results_dict_{k_id}.pkl')
        if os.path.exists(out_file) and not args.force_overwrite:
            continue

        try:
            if 'science' in dataset_name:
                question = sample['question']
                solution = sample.get('solution', '')
                choices = sample.get('choices', [])
                ans_idx = sample.get('answer', -1)

                correct_answers = []
                if solution:
                    correct_answers.append(solution)
                if choices and 0 <= ans_idx < len(choices):
                    correct_choice = choices[ans_idx]
                    if correct_choice not in correct_answers:
                        correct_answers.append(correct_choice)
                if not correct_answers:
                    correct_answers = [""]
                best_answer = correct_answers[0]

                incorrect_answers = []
                if choices:
                    for c_idx, choice in enumerate(choices):
                        if c_idx != ans_idx:
                            incorrect_answers.append(choice)

                if choices:
                    formatted_choices = ", ".join(choices)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. Based on the given choices, answer the question in a single sentence. Do not repeat the prompt."},
                        {"role": "user", "content": f"Choices: {formatted_choices}\nQuestion: {question}"}
                    ]
                    fallback_text = f"Choices: {formatted_choices}\nQuestion: {question}\nAnswer:"
                else:
                    messages = [
                        {"role": "system", "content": "Answer the user's question in a single sentence. Do not repeat the prompt."},
                        {"role": "user", "content": question}
                    ]
                    fallback_text = f"Question: {question}\nAnswer:"

            elif 'open' in dataset_name:
                question = sample.get('question_stem', sample.get('question', ''))
                choices = sample.get('choices', {})
                choices_text = choices.get('text', []) if isinstance(choices, dict) else (choices if isinstance(choices, list) else [])
                choices_labels = choices.get('label', []) if isinstance(choices, dict) else [chr(ord('A') + i) for i in range(len(choices_text))]
                formatted_choices = ", ".join([f"{l}: {t}" for l, t in zip(choices_labels, choices_text)]) if choices_text else ""

                ans_key = sample.get('answerKey', sample.get('answer', ''))
                ans_idx = -1
                if isinstance(ans_key, str) and ans_key in choices_labels:
                    ans_idx = choices_labels.index(ans_key)
                elif isinstance(ans_key, int):
                    ans_idx = ans_key

                correct_choice = ""
                if choices_text and 0 <= ans_idx < len(choices_text):
                    correct_choice = choices_text[ans_idx]

                correct_answers = []
                if correct_choice:
                    correct_answers.append(correct_choice)
                if not correct_answers:
                    correct_answers = [str(ans_key)] if ans_key else [""]
                best_answer = correct_answers[0]

                incorrect_answers = []
                if choices_text:
                    for c_idx, choice in enumerate(choices_text):
                        if c_idx != ans_idx:
                            incorrect_answers.append(choice)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Based on the given choices, answer the question in a single sentence. Do not repeat the prompt."},
                    {"role": "user", "content": f"Choices: {formatted_choices}\nQuestion: {question}"}
                ]
                fallback_text = f"Choices: {formatted_choices}\nQuestion: {question}\nAnswer:"

            elif 'coqa' in dataset_name:
                question = sample.get('questions', [''])[0] if isinstance(sample.get('questions'), list) else sample.get('question', '')
                story = sample.get('story', '')
                answers = sample.get('answers', {})
                correct_answers = answers.get('input_text', []) if isinstance(answers, dict) else [str(answers)]
                if not correct_answers:
                    correct_answers = [""]
                best_answer = correct_answers[0]
                incorrect_answers = []

                user_content = f"Story: {story}\nQuestion: {question}" if story else question
                messages = [
                    {"role": "system", "content": "Provide a direct, concise, single-sentence answer to the user's question based on the story. Do not explain your answer."},
                    {"role": "user", "content": user_content}
                ]
                fallback_text = f"Story: {story}\nQuestion: {question}\nAnswer:" if story else f"Question: {question}\nAnswer:"

            else:  # truthful_qa / truthfulqa
                question = sample['question']
                best_answer = sample['best_answer']
                correct_answers = sample.get('correct_answers', [best_answer])
                incorrect_answers = sample.get('incorrect_answers', [])

                messages = [
                    {"role": "system", "content": "Provide a direct, concise, single-sentence answer to the user's question. Do not repeat system instructions, do not repeat the prompt, do not list multiple options, and do not explain your answer."},
                    {"role": "user", "content": question}
                ]
                fallback_text = f"Answer the following question directly and concisely in a single sentence:\nQuestion: {question}\nAnswer:"

            try:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = fallback_text

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device_llm)
            prompt_ids = inputs["input_ids"][0].to('cpu')
            detached_input_ids = inputs['input_ids'].detach().to('cpu')

            # 1. Deterministic Greedy Generation
            most_likely_gen = generate_text(
                args=args,
                model=llm_model,
                tokenizer=tokenizer,
                input_ids=inputs["input_ids"],
                len_prompt=len(prompt_ids),
                decoding_method='most_likely',
                device=device_llm
            )

            # 2. Likelihood of initial generation
            most_likely_likelihoods = compute_likelihood(
                prompt=prompt_ids,
                generation=most_likely_gen,
                model=llm_model,
                device=device_llm,
                compute_cleaned=args.compute_cleaned,
                store_logits=True
            )

            # 3. Efficient all-layers hidden states evaluation
            epistemic_entropy_list = evaluate_all_layers_for_sample(
                args=args,
                model=llm_model,
                tokenizer=tokenizer,
                deberta_model=deberta_model,
                deberta_tokenizer=deberta_tokenizer,
                deberta_embeddings=deberta_embeddings,
                device_llm=device_llm,
                device_deberta=device_deberta,
                input_ids=inputs['input_ids'],
                prompt=prompt_ids,
                question=question,
                initial_generation=most_likely_gen,
                initial_likelihood=most_likely_likelihoods
            )

            results_dict = {
                'input_ids': detached_input_ids,
                'question': question,
                'question_id': idx,
                'choices': choices if 'choices' in locals() and choices else [],
                'best_answer': best_answer,
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers,
                'generated_text': most_likely_gen['generation_text'][0],
                'correctness_dict': {},  # Evaluated in batch phase
                'sdlg': {
                    'generations': [most_likely_gen],
                    'likelihoods': [most_likely_likelihoods],
                    'epistem_entropies': epistemic_entropy_list
                }
            }

            if args.compress_pkl:
                with gzip.open(out_file, 'wb') as f:
                    pickle.dump(results_dict, f)
            else:
                with open(out_file, 'wb') as f:
                    pickle.dump(results_dict, f)

            # Real-time console logging
            ans_text = most_likely_gen['generation_text'][0]
            summary_metrics = epistemic_entropy_list[0] if epistemic_entropy_list else {}
            h_mix_val = summary_metrics.get('entropy_mixture', 0.0)
            mi_val = summary_metrics.get('mutual_info', 0.0)
            gmi_val = summary_metrics.get('gated_mutual_info', 0.0)
            mi75_val = summary_metrics.get('mutual_info_depth_75', 0.0)
            gmi75_val = summary_metrics.get('gated_mutual_info_depth_75', 0.0)

            tqdm.write("\n" + "=" * 60)
            tqdm.write(f"Sample #{idx + 1}/{num_samples} [ID: {idx}]")
            tqdm.write(f"Question:  {question}")
            if ('science' in dataset_name or 'open' in dataset_name) and choices:
                tqdm.write(f"Choices:   {choices}")
            tqdm.write(f"Generated: '{ans_text}'")
            tqdm.write(f"Metrics:   H_mix={h_mix_val:.4f} | Original_MI={mi_val:.4f} | Gated_MI={gmi_val:.4f}")
            tqdm.write(f"75% Depth: MI_75={mi75_val:.4f} | Gated_MI_75={gmi75_val:.4f}")
            tqdm.write("=" * 60)

        except Exception as e:
            print(f"[ERROR] Sample {k_id}: {e}")
            traceback.print_exc()

    # =========================================================================
    # PHASE 2: FREE PYTORCH LLM VRAM TO AVOID TF / BLEURT CONFLICTS
    # =========================================================================
    print("\n[PHASE 2] Generation complete. Freeing PyTorch LLM GPU memory...")
    del llm_model, deberta_model, deberta_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 3: VECTORIZED BATCH BLEURT SCORING
    # =========================================================================
    print("\n[PHASE 3] Loading BLEURT (bleurt-base-128) for vectorized batched scoring...")
    
    # TensorFlow Memory Allocation Guard
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    import evaluate
    try:
        bleurt = evaluate.load('bleurt', 'bleurt-base-128')
    except Exception:
        try:
            bleurt = evaluate.load('bleurt')
        except Exception:
            print("[WARNING] BLEURT could not be loaded.")
            bleurt = None

    # Load all generated files
    files_gz = glob.glob(os.path.join(base_path, "results_dict_*.pkl.gz"))
    files_raw = glob.glob(os.path.join(base_path, "results_dict_*.pkl"))
    all_files = sorted(files_gz + files_raw, key=lambda p: int(os.path.basename(p).replace("results_dict_", "").split(".")[0]))

    print(f"[INFO] Loaded {len(all_files)} sample files. Preparing batch BLEURT evaluation...")

    loaded_samples = []
    pairs_for_bleurt = []

    for fpath in tqdm(all_files, desc="Phase 3: Indexing Samples"):
        try:
            is_gz = fpath.endswith('.gz')
            if is_gz:
                with gzip.open(fpath, 'rb') as f:
                    res = pickle.load(f)
            else:
                with open(fpath, 'rb') as f:
                    res = pickle.load(f)

            gen_text = res.get('generated_text', res['sdlg']['generations'][0]['generation_text'][0])
            ref_answers = res.get('correct_answers', [res.get('best_answer', '')])
            q_text = res.get('question', '')
            c_dict = res.get('correctness_dict', {})
            s_idx = len(loaded_samples)

            if 'bleurt' not in c_dict and bleurt is not None:
                pairs_for_bleurt.append((s_idx, gen_text, ref_answers))

            loaded_samples.append({
                'fpath': fpath,
                'is_gz': is_gz,
                'res': res,
                'gen_text': gen_text,
                'ref_answers': ref_answers,
                'q_text': q_text,
                'c_dict': c_dict
            })
        except Exception as e:
            print(f"[WARNING] Sample read error {fpath}: {e}")

    # Compute BLEURT across all samples in vectorized batches
    bleurt_batch_size = getattr(args, 'bleurt_batch_size', 64)
    computed_bleurt_map = compute_batched_bleurt(bleurt, pairs_for_bleurt, batch_size=bleurt_batch_size)

    all_bleurt_scores = []
    all_batch_questions = []
    all_semantic_entropies = []

    for s_idx, item in enumerate(loaded_samples):
        res = item['res']
        c_dict = item['c_dict']
        if s_idx in computed_bleurt_map:
            c_dict['bleurt'] = computed_bleurt_map[s_idx]
            res['correctness_dict'] = c_dict
            # Re-save file with computed BLEURT score
            if item['is_gz']:
                with gzip.open(item['fpath'], 'wb') as f:
                    pickle.dump(res, f)
            else:
                with open(item['fpath'], 'wb') as f:
                    pickle.dump(res, f)

        b_score = c_dict.get('bleurt', 0.0)
        all_bleurt_scores.append(b_score)
        all_batch_questions.append(item['q_text'])

        epistemic_list = res.get('sdlg', {}).get('epistem_entropies', [])
        if epistemic_list and isinstance(epistemic_list, list) and len(epistemic_list) > 0:
            all_keys = epistemic_list[0].keys()
            avg_sample_dict = {}
            for k in all_keys:
                vals = [step[k] for step in epistemic_list if k in step and isinstance(step[k], (int, float))]
                if vals:
                    avg_sample_dict[k] = sum(vals) / len(vals)
            all_semantic_entropies.append(avg_sample_dict)
        else:
            all_semantic_entropies.append({})

    # =========================================================================
    # PHASE 4: PERSIST TO RESULT/ENTROPIES3/ FOR DOWNSTREAM NOTEBOOKS
    # =========================================================================
    out_label_file = os.path.join(base_path_entropies, f"ile_label_{data_tag}_{modl_tag_with_quant}_new.pkl")
    out_question_file = os.path.join(base_path_entropies, f"ile_question_{data_tag}_{modl_tag_with_quant}_new.pkl")
    out_entropies_file = os.path.join(base_path_entropies, f"ile_entropies_{data_tag}_{modl_tag_with_quant}_new.pkl")

    legacy_label = os.path.join(base_path_entropies, f"sdlg_label_{data_tag}_{modl_tag_with_quant}_new.pkl")
    legacy_question = os.path.join(base_path_entropies, f"sdlg_question_{data_tag}_{modl_tag_with_quant}_new.pkl")
    legacy_entropies = os.path.join(base_path_entropies, f"sdlg_entropies_{data_tag}_{modl_tag_with_quant}_new.pkl")

    for fpath in [out_label_file, legacy_label]:
        with open(fpath, 'wb') as f:
            pickle.dump(all_bleurt_scores, f)
    for fpath in [out_question_file, legacy_question]:
        with open(fpath, 'wb') as f:
            pickle.dump(all_batch_questions, f)
    for fpath in [out_entropies_file, legacy_entropies]:
        with open(fpath, 'wb') as f:
            pickle.dump(all_semantic_entropies, f)

    print("\n" + "=" * 80)
    print("ALL-LAYERS EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Total Samples Processed: {len(all_bleurt_scores)}")
    print(f"Saved Raw Pickles:        {base_path}/")
    print(f"Saved Entropies3 Labels:  {out_label_file}")
    print(f"Saved Entropies3 Qs:      {out_question_file}")
    print(f"Saved Entropies3 Metrics: {out_entropies_file}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    run_pipeline()
