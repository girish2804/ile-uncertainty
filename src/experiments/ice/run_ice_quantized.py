import os
import sys

# Prevent TensorFlow (used by BLEURT) from hogging GPU VRAM needed by 70B quantized LLM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# DynamicCache Compatibility Shim for modern HuggingFace Transformers
try:
    from transformers.cache_utils import DynamicCache, Cache
    
    def _get_seen_tokens(self, layer_idx=0):
        if hasattr(self, "_seen_tokens_val") and self._seen_tokens_val is not None:
            return self._seen_tokens_val
        try:
            return self.get_seq_length(layer_idx)
        except Exception:
            try:
                return self.get_seq_length()
            except Exception:
                return len(self.key_cache[0][0]) if (hasattr(self, 'key_cache') and self.key_cache) else 0

    def _set_seen_tokens(self, val):
        self._seen_tokens_val = val

    DynamicCache.seen_tokens = property(_get_seen_tokens, _set_seen_tokens)
    DynamicCache._seen_tokens = property(_get_seen_tokens, _set_seen_tokens)

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

import gc
import math
import random
import pickle
import gzip
try:
    import yaml
except ImportError:
    yaml = None
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import evaluate
import traceback

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

# Add current directory to sys.path for importing ice_quantized_args
ice_dir = os.path.dirname(os.path.abspath(__file__))
if ice_dir not in sys.path:
    sys.path.append(ice_dir)

from ice_quantized_args import ICEQuantizedArgs


def seed_everything(seed):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class StopOnTokens(StoppingCriteria):
    """Stops LLM GPU generation immediately when a question mark, newline, or prompt marker is generated in newly generated tokens."""
    def __init__(self, tokenizer, prompt_len, stop_strings=["?", "\n", "Original Question"]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0][self.prompt_len:]
        if len(gen_ids) <= 1:
            return False
        latest_ids = gen_ids[-16:]
        decoded_text = self.tokenizer.decode(latest_ids, skip_special_tokens=True)
        return any(stop in decoded_text for stop in self.stop_strings)


def resolve_eos_ids(tokenizer, model):
    """Dynamically resolves EOS and special stop tokens for modern LLM architectures (LLaMA-3/3.1, Qwen2.5, Phi-3)."""
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)
    if hasattr(model, 'config') and model.config.eos_token_id is not None:
        if isinstance(model.config.eos_token_id, list):
            eos_ids.extend(model.config.eos_token_id)
        else:
            eos_ids.append(model.config.eos_token_id)
    for token_str in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|end|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    return list(set(eos_ids))


@torch.no_grad()
def generate_sequence(model, tokenizer, prompt_str, do_sample, temperature, top_p, num_beams, max_new_tokens, device, stop_on_question=False, eos_token_ids=None):
    """
    Direct HuggingFace generation with robust device resolution and text extraction.
    """
    if hasattr(model, 'device'):
        target_device = model.device
    else:
        try:
            target_device = next(model.parameters()).device
        except Exception:
            target_device = device
    input_ids = tokenizer.encode(prompt_str, return_tensors='pt').to(target_device)
    prompt_len = input_ids.shape[1]
    
    pad_id_gen = tokenizer.pad_token_id
    if pad_id_gen is None or pad_id_gen == tokenizer.eos_token_id:
        pad_id_gen = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

    if eos_token_ids is not None and len(eos_token_ids) > 0:
        eos_id = eos_token_ids
    else:
        eos_id = getattr(model.config, 'eos_token_id', tokenizer.eos_token_id)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id_gen,
        "eos_token_id": eos_id
    }

    if stop_on_question:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(tokenizer, prompt_len=prompt_len, stop_strings=["?", "\n", "Original Question"])])

    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": max(temperature, 0.01),
            "top_p": top_p
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": max(num_beams, 1)
        })

    output_ids = model.generate(input_ids, **gen_kwargs)[0]
    gen_ids = output_ids[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if "</think>" in gen_text:
        gen_text = gen_text.split("</think>")[-1].strip()

    if stop_on_question:
        clean_text = gen_text.split('\n')[0].split('Original Question')[0].split('Clarified Question')[0].strip()
        if '?' in clean_text:
            clean_text = clean_text.split('?')[0] + '?'
        clean_text = clean_text.strip()
    else:
        first_line = gen_text.split('\n')[0].strip()
        clean_text = first_line.split('Original Question')[0].split('Clarified Question')[0].strip()
        if clean_text.startswith('A:'):
            clean_text = clean_text[2:].strip()
        elif clean_text.startswith('Answer:'):
            clean_text = clean_text[7:].strip()
        elif clean_text.startswith('Q:'):
            clean_text = clean_text[2:].strip()
        clean_text = clean_text.strip()

        if set(clean_text).issubset({'.', ' ', ':', '-'}) or len(clean_text) == 0:
            clean_text = gen_text.strip()

    if clean_text != gen_text and len(clean_text) > 0:
        gen_text = clean_text
        gen_ids = tokenizer.encode(gen_text, add_special_tokens=False, return_tensors='pt')[0]

    return {
        'prompt_ids': input_ids[0].cpu(),
        'generation_ids': gen_ids.cpu(),
        'generation_text': gen_text
    }


@torch.no_grad()
def compute_nll(model, prompt_ids, gen_ids, device):
    """
    Computes average negative log-likelihood (NLL) per token for a generated sequence.
    Handles sharded / auto-mapped models cleanly.
    """
    if len(gen_ids) == 0:
        return 0.0

    if hasattr(model, 'device'):
        target_device = model.device
    else:
        try:
            target_device = next(model.parameters()).device
        except Exception:
            target_device = device
    full_ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(target_device)
    labels = full_ids.clone()
    labels[0, :len(prompt_ids)] = -100

    outputs = model(full_ids, labels=labels)
    return outputs.loss.item()


def compute_nli_semantic_clusters(gen_texts, deberta_tokenizer, deberta_model, device_deberta):
    """
    Computes semantic clustering using DeBERTa-MNLI bidirectional entailment directly,
    matching the official ICE repository and ambig.ipynb methodology.
    """
    num_gens = len(gen_texts)
    if num_gens <= 1:
        return torch.zeros(num_gens, dtype=torch.long)

    adj_matrix = np.eye(num_gens, dtype=bool)
    pairs = []
    indices = []

    for i in range(num_gens):
        for j in range(num_gens):
            if i == j:
                continue
            pairs.append((gen_texts[i], gen_texts[j]))
            indices.append((i, j))

    if pairs:
        batch_size = 16
        entailment_mask = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = deberta_tokenizer(
                [p[0] for p in batch_pairs],
                [p[1] for p in batch_pairs],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device_deberta)
            with torch.no_grad():
                logits = deberta_model(**inputs).logits
                preds = torch.argmax(logits, dim=1)
            # Label 2 = Entailment in DeBERTa-MNLI
            entailment_mask.extend((preds == 2).cpu().tolist())

        for idx, is_entailed in enumerate(entailment_mask):
            i, j = indices[idx]
            if is_entailed:
                adj_matrix[i, j] = True

    # Bidirectional entailment clustering
    cluster_mapping = [-1] * num_gens
    next_cluster_id = 0
    for i in range(num_gens):
        if cluster_mapping[i] != -1:
            continue
        cluster_mapping[i] = next_cluster_id
        for j in range(i + 1, num_gens):
            if cluster_mapping[j] == -1:
                if adj_matrix[i, j] and adj_matrix[j, i]:
                    cluster_mapping[j] = next_cluster_id
        next_cluster_id += 1

    return torch.tensor(cluster_mapping, dtype=torch.long)


def compute_ice_uncertainties(gen_texts, nlls, deberta_model, deberta_tokenizer, device_deberta):
    """
    Computes exact ICE uncertainties (Hou et al. 2024 / UCSB-NLP-Chang):
    1. Epistemic Uncertainty (EU): Average expected entropy across clarifications E_c[H(Y|X, c)]
    2. Total Uncertainty (TU): Marginal mixture entropy H(Y|X) across all pooled responses
    3. Aleatoric Uncertainty (AU): TU - EU (Exact unclamped subtraction from official ICE repo)
    """
    num_gens = len(gen_texts)
    if num_gens == 0:
        return {'epistemic': 0.0, 'total': 0.0, 'aleatoric': 0.0, 'semantic_entropy_norm': 0.0, 'num_clusters': 0}

    # 1. Epistemic Uncertainty: Average individual NLL per response
    epistemic_uncertainty = float(np.mean(nlls)) if len(nlls) > 0 else 0.0

    # 2. Total Uncertainty: DeBERTa-MNLI Semantic Entropy across clarification responses
    cluster_ids = compute_nli_semantic_clusters(
        gen_texts=gen_texts,
        deberta_tokenizer=deberta_tokenizer,
        deberta_model=deberta_model,
        device_deberta=device_deberta
    )
    
    # Compute mixture weights from NLLs
    weights = torch.exp(-torch.tensor(nlls, dtype=torch.float32))
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = torch.ones(num_gens, dtype=torch.float32) / num_gens

    # Aggregate probabilities per semantic cluster
    unique_clusters = torch.unique(cluster_ids)
    num_unique = len(unique_clusters)
    cluster_probs = torch.zeros(num_unique, dtype=torch.float32)
    for idx, c_id in enumerate(unique_clusters):
        cluster_probs[idx] = weights[cluster_ids == c_id].sum()

    # Unnormalized & Normalized Shannon Entropy
    total_uncertainty = float(-torch.sum(cluster_probs * torch.log(cluster_probs + 1e-12)).item())
    norm_factor = math.log(num_gens) if num_gens > 1 else 1.0
    semantic_entropy_norm = total_uncertainty / norm_factor

    # 3. Aleatoric Uncertainty: Exact unclamped subtraction as implemented in official ICE codebase
    aleatoric_uncertainty = total_uncertainty - epistemic_uncertainty

    return {
        'epistemic': epistemic_uncertainty,
        'total': total_uncertainty,
        'aleatoric': aleatoric_uncertainty,
        'semantic_entropy_norm': semantic_entropy_norm,
        'num_clusters': int(num_unique)
    }


def save_cache_atomically(cache_dict, target_filepath):
    """Saves dictionary atomically using a temporary file to prevent corruption if interrupted."""
    os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
    tmp_filepath = target_filepath + ".tmp"
    with open(tmp_filepath, 'wb') as f:
        pickle.dump(cache_dict, f)
    os.replace(tmp_filepath, target_filepath)


def format_answer_prompt(tokenizer, question, choices=None, dataset_name='truthfulqa'):
    """
    Constructs prompt adhering to exact prompt consistency standards across ILE and baselines.
    Uses chat templates when supported, otherwise falls back to standardized structured strings.
    """
    clean_d = dataset_name.lower().replace("_", "").replace("-", "")

    if choices:
        formatted_choices = ", ".join(choices) if isinstance(choices, list) else str(choices)
        user_content = f"Choices: {formatted_choices}\nQuestion: {question}"
        system_content = (
            "You are a helpful assistant. Based on the given choices, answer the question "
            "in a single sentence. Do not repeat the prompt."
        )
        fallback_text = f"Choices: {formatted_choices}\nQuestion: {question}\nAnswer:"
    else:
        user_content = question
        system_content = (
            "Answer the user's question in a single sentence. Do not repeat the prompt."
        )
        fallback_text = f"Answer the following question directly and concisely in a single sentence:\nQuestion: {question}\nAnswer:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return fallback_text


def extract_sample_data(sample, dataset_name='truthfulqa'):
    """
    Extracts question, choices, and correct reference answers consistently across datasets.
    """
    clean_d = dataset_name.lower().replace('_', '').replace('-', '')

    if 'truth' in clean_d:
        question = sample['question']
        best_answer = sample.get('best_answer', '')
        correct_answers = sample.get('correct_answers', [])
        if isinstance(correct_answers, list) and len(correct_answers) > 0:
            ref_ans = [best_answer] + [a for a in correct_answers if a != best_answer] if best_answer else correct_answers
        elif best_answer:
            ref_ans = [best_answer]
        else:
            ref_ans = [""]

        return {
            'question': question,
            'choices': None,
            'reference_answers': ref_ans
        }

    elif 'open' in clean_d:
        question = sample.get('question_stem', sample.get('question', ''))
        raw_choices = sample.get('choices', {})
        if isinstance(raw_choices, dict):
            choices_text = raw_choices.get('text', [])
            choices_labels = raw_choices.get('label', [chr(ord('A') + i) for i in range(len(choices_text))])
        elif isinstance(raw_choices, list):
            choices_text = raw_choices
            choices_labels = [chr(ord('A') + i) for i in range(len(choices_text))]
        else:
            choices_text = []
            choices_labels = []

        ans_key = sample.get('answerKey', sample.get('answer', ''))
        ans_idx = -1
        if isinstance(ans_key, str) and ans_key in choices_labels:
            ans_idx = choices_labels.index(ans_key)
        elif isinstance(ans_key, int):
            ans_idx = ans_key

        correct_choice = ""
        if choices_text and 0 <= ans_idx < len(choices_text):
            correct_choice = str(choices_text[ans_idx]).strip()

        fact1 = sample.get('fact1', '')
        correct_answers = []
        if correct_choice:
            correct_answers.append(correct_choice)
        if fact1 and str(fact1).strip() and str(fact1).strip() not in correct_answers:
            correct_answers.append(str(fact1).strip())
        if not correct_answers:
            correct_answers = [str(ans_key)] if ans_key else [""]

        return {
            'question': question,
            'choices': choices_text,
            'reference_answers': correct_answers
        }

    elif 'science' in clean_d or 'sciq' in clean_d:
        question = sample['question']
        solution = sample.get('solution', '')
        choices = sample.get('choices', [])
        ans_idx = sample.get('answer', -1)

        correct_answers = []
        if choices and isinstance(choices, list) and 0 <= ans_idx < len(choices):
            correct_choice = str(choices[ans_idx]).strip()
            correct_answers.append(correct_choice)
        if solution and isinstance(solution, str) and solution.strip():
            sol_str = solution.strip()
            if sol_str not in correct_answers:
                correct_answers.append(sol_str)
        if not correct_answers:
            correct_answers = [str(ans_idx)] if ans_idx != -1 else [""]

        return {
            'question': question,
            'choices': choices,
            'reference_answers': correct_answers
        }

    else:
        question = sample.get('question', '')
        ans = sample.get('answer', '')
        return {
            'question': question,
            'choices': None,
            'reference_answers': [str(ans)] if ans else [""]
        }


def resolve_clarification_cache_file(script_dir, dataset_clean, clar_model_clean):
    """
    Resolves clarification cache file location, checking both 'cache/' subdirectory and script directory.
    """
    cache_dir = os.path.join(script_dir, 'cache')
    primary_file = os.path.join(cache_dir, f"clarifications_{dataset_clean}_{clar_model_clean}.pkl")
    
    candidates = [
        primary_file,
        primary_file + ".gz",
        os.path.join(script_dir, f"clarifications_{dataset_clean}_{clar_model_clean}.pkl"),
        os.path.join(script_dir, f"clarifications_{dataset_clean}_{clar_model_clean}.pkl.gz"),
        # Also check with hyphenated variations
        os.path.join(cache_dir, f"clarifications_{dataset_clean.replace('_', '')}_{clar_model_clean}.pkl"),
        os.path.join(script_dir, f"clarifications_{dataset_clean.replace('_', '')}_{clar_model_clean}.pkl")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return primary_file


def run_ice_quantized_experiment():
    args = ICEQuantizedArgs()
    num_clarifications = getattr(args, 'num_clarifications', 5)

    model_name_clean = args.llm_model.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "").replace("_", "-")
    dataset_clean = getattr(args, 'dataset', 'truthfulqa').lower().replace("_", "").replace("-", "")
    quant_suffix = f"_{args.quantization}" if getattr(args, 'quantization', None) else ""
    args.run_id = f"ice-{dataset_clean}-{model_name_clean}{quant_suffix}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'result', args.run_id)
    os.makedirs(base_path, exist_ok=True)

    # Save run arguments
    args.args_to_yaml(base_path)

    print("=" * 70)
    print(f"RUNNING QUANTIZED ICE BASELINE EXPERIMENT: {args.run_id}")
    print(f"Dataset                  : {args.dataset}")
    print(f"LLM (Quantized Target)   : {args.llm_model}")
    print(f"Quantization Mode        : {args.quantization} (BitsAndBytes NF4/bfloat16)")
    print(f"Attention Implementation : {getattr(args, 'attn_implementation', 'sdpa')}")
    print(f"Clarifications Generator : {args.clarification_model}")
    print(f"Clarifications per sample: {num_clarifications}")
    print(f"Results Directory        : {base_path}")
    print("=" * 70)

    seed_everything(seed=args.seed_value)

    device_llm = "cuda" if torch.cuda.is_available() else "cpu"
    target_deberta_dev = getattr(args, 'deberta_device', 'cuda')
    device_deberta = target_deberta_dev if (target_deberta_dev != 'cuda' or torch.cuda.is_available()) else 'cpu'
    auth_token = getattr(args, 'hf_token', None)

    # 1. Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    clean_dataset = dataset_clean
    if 'truth' in clean_dataset:
        try:
            raw_dataset = load_dataset('truthfulqa/truthful_qa', 'generation')['validation']
        except Exception:
            raw_dataset = load_dataset('truthful_qa', 'generation')['validation']
    elif 'open' in clean_dataset:
        try:
            raw_dataset = load_dataset('allenai/openbookqa', 'additional')['test']
        except Exception:
            raw_dataset = load_dataset('allenai/openbookqa', 'main')['test']
    elif 'science' in clean_dataset or 'sciq' in clean_dataset:
        try:
            raw_dataset = load_dataset('tasksource/ScienceQA_text_only')['test']
        except Exception:
            try:
                raw_dataset = load_dataset('allenai/sciq')['test']
            except Exception:
                raw_dataset = load_dataset('derek-thomas/ScienceQA')['test']
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Add question_id column if missing
    if 'question_id' not in raw_dataset.column_names:
        raw_dataset = raw_dataset.add_column('question_id', list(range(len(raw_dataset))))

    # Slice requested sample range
    end_idx = len(raw_dataset) if args.num_samples == -1 else min(args.start_idx + args.num_samples, len(raw_dataset))
    dataset_subset = raw_dataset.select(range(args.start_idx, end_idx))

    # 2. Check & Pre-populate Clarification Cache (Two-Phase Memory Protection for 70B)
    clar_model_name = getattr(args, 'clarification_model', 'meta-llama/Meta-Llama-3-8B-Instruct')
    clar_model_clean = clar_model_name.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "")
    cache_file = resolve_clarification_cache_file(script_dir, dataset_clean, clar_model_clean)

    clarification_cache = {}
    if getattr(args, 'cache_clarifications', True) and os.path.exists(cache_file):
        try:
            if cache_file.endswith('.gz'):
                with gzip.open(cache_file, 'rb') as f:
                    clarification_cache = pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    clarification_cache = pickle.load(f)
            print(f"Loaded existing clarification cache ({len(clarification_cache)} samples) from {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to load clarification cache from {cache_file}: {e}")
            clarification_cache = {}

    # Check for missing clarifications in requested range
    missing_samples = []
    for s in dataset_subset:
        s_key = str(int(s['question_id']))
        if s_key not in clarification_cache or len(clarification_cache[s_key]) < num_clarifications:
            missing_samples.append(s)

    # Phase 1: If any clarifications are missing, generate them standalone before loading the 70B model!
    if missing_samples:
        print(f"\n[Phase 1: Clarification Pre-caching] Found {len(missing_samples)} samples needing clarifications.")
        print(f"Loading clarification generator model standalone: {args.clarification_model}...")
        is_clar_phi = "phi-3" in str(args.clarification_model).lower() or "phi3" in str(args.clarification_model).lower()
        clar_trust_remote = False if is_clar_phi else True
        try:
            clar_tokenizer = AutoTokenizer.from_pretrained(args.clarification_model, token=auth_token, trust_remote_code=clar_trust_remote)
        except Exception as err_tok:
            print(f"Fast tokenizer loading failed ({err_tok}). Retrying with use_fast=False...")
            clar_tokenizer = AutoTokenizer.from_pretrained(args.clarification_model, token=auth_token, use_fast=False, trust_remote_code=clar_trust_remote)

        # Use 4-bit quantization on clarification model as well to conserve GPU VRAM during pre-caching
        clar_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ) if torch.cuda.is_available() else None

        if torch.cuda.is_available():
            clar_model = AutoModelForCausalLM.from_pretrained(
                args.clarification_model,
                quantization_config=clar_bnb_config,
                torch_dtype=torch.bfloat16,
                token=auth_token,
                trust_remote_code=clar_trust_remote,
                device_map="auto"
            )
        else:
            clar_model = AutoModelForCausalLM.from_pretrained(
                args.clarification_model,
                torch_dtype=torch.float32,
                token=auth_token,
                trust_remote_code=clar_trust_remote
            ).to(device_llm)

        if clar_tokenizer.pad_token is None:
            clar_tokenizer.pad_token = clar_tokenizer.eos_token if clar_tokenizer.eos_token is not None else clar_tokenizer.bos_token
        clar_model.config.pad_token_id = clar_tokenizer.pad_token_id
        c_eos_ids = resolve_eos_ids(clar_tokenizer, clar_model)

        print(f"Generating missing clarifications for {len(missing_samples)} samples...")
        for sample in tqdm(missing_samples, desc="Phase 1: Generating Clarifications"):
            sample_idx = int(sample['question_id'])
            sample_key = str(sample_idx)
            sample_data = extract_sample_data(sample, args.dataset)
            q_text = sample_data['question']
            eff_choices = sample_data['choices'] if getattr(args, 'use_options', False) else None

            if getattr(args, 'use_icl_prompt', True):
                if eff_choices:
                    formatted_choices = ", ".join(eff_choices) if isinstance(eff_choices, list) else str(eff_choices)
                    clar_prompt = (
                        "Below is a multiple-choice question that may be ambiguous. Please generate a clarified and unambiguous version of the question that still asks for one of the given choices.\n\n"
                        "Choices: France, Argentina, Brazil\n"
                        "Original Question: Who won the world cup?\n"
                        "Clarified Question: Which national team won the FIFA Men's World Cup in 2022?\n\n"
                        "Choices: Pacific Northwest, East Coast, Capital\n"
                        "Original Question: Where is Washington?\n"
                        "Clarified Question: What is the geographic location of Washington State in the US?\n\n"
                        f"Choices: {formatted_choices}\n"
                        f"Original Question: {q_text}\n"
                        "Clarified Question:"
                    )
                else:
                    clar_prompt = (
                        "Below is a potentially ambiguous question. Please generate a clarified and unambiguous version of the question.\n\n"
                        "Original Question: Who won the world cup?\n"
                        "Clarified Question: Which national team won the FIFA Men's World Cup in 2022?\n\n"
                        "Original Question: Where is Washington?\n"
                        "Clarified Question: What is the geographic location of Washington State in the US?\n\n"
                        f"Original Question: {q_text}\n"
                        "Clarified Question:"
                    )
            else:
                if eff_choices:
                    formatted_choices = ", ".join(eff_choices) if isinstance(eff_choices, list) else str(eff_choices)
                    clar_prompt = f"Given choices [{formatted_choices}], a much clearer and unambiguous way to ask the question '{q_text}' is: "
                else:
                    clar_prompt = f"If someone asks the ambiguous question '{q_text}', a much clearer and unambiguous way to ask it is: "

            generated_clarifications = []
            for _ in range(num_clarifications):
                clar_gen = generate_sequence(
                    model=clar_model,
                    tokenizer=clar_tokenizer,
                    prompt_str=clar_prompt,
                    do_sample=args.do_sample_baseline,
                    temperature=args.temperature_baseline,
                    top_p=args.top_p_baseline,
                    num_beams=args.num_beams_baseline,
                    max_new_tokens=args.max_length_of_generated_sequence,
                    device=device_llm,
                    stop_on_question=True,
                    eos_token_ids=c_eos_ids
                )
                if len(clar_gen['generation_text']) > 0:
                    generated_clarifications.append(clar_gen['generation_text'])

            if len(generated_clarifications) == 0:
                generated_clarifications = [q_text]

            clarification_cache[sample_key] = generated_clarifications

        save_cache_atomically(clarification_cache, cache_file)
        print(f"Saved updated clarification cache ({len(clarification_cache)} total samples) to {cache_file}")

        # Explicitly unload clarification model and free GPU memory before loading 70B model!
        del clar_model
        del clar_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Phase 1 Complete] Clarification generator unloaded and GPU VRAM freed.\n")
    else:
        print(f"[Clarification Cache] 100% cache hit ({len(dataset_subset)} samples available). No clarification model loading needed!\n")

    # 3. Phase 2: Load 70B Quantized Target LLM FIRST (to dedicate 100% GPU VRAM to the large model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading 70B Target LLM: {args.llm_model}...")
    is_phi = "phi-3" in str(args.llm_model).lower() or "phi3" in str(args.llm_model).lower()
    trust_remote = False if is_phi else True
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, token=auth_token, trust_remote_code=trust_remote)
    except Exception as err_tok:
        print(f"Fast tokenizer loading failed ({err_tok}). Retrying with use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, token=auth_token, use_fast=False, trust_remote_code=trust_remote)

    from transformers import AutoConfig, PretrainedConfig
    print(f"Loading LLM Config: {args.llm_model}...")
    try:
        config = AutoConfig.from_pretrained(args.llm_model, token=auth_token, trust_remote_code=trust_remote)
    except Exception as err_cfg:
        print(f"Notice: Standard AutoConfig loading failed ({err_cfg}). Attempting config patch fallback...")
        try:
            config_dict, _ = PretrainedConfig.get_config_dict(args.llm_model, token=auth_token)
            if "rope_scaling" in config_dict and isinstance(config_dict["rope_scaling"], dict):
                rs = config_dict["rope_scaling"]
                if "rope_type" in rs or "high_freq_factor" in rs or len(rs) > 2:
                    config_dict["rope_scaling"] = {
                        "type": "linear",
                        "factor": float(rs.get("factor", 32.0))
                    }
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            model_type = config_dict.get("model_type", "llama")
            if model_type in CONFIG_MAPPING:
                config = CONFIG_MAPPING[model_type].from_dict(config_dict)
            else:
                config_dict.pop("model_type", None)
                config = AutoConfig.for_model(model_type, **config_dict)
        except Exception as err_patch:
            raise RuntimeError(f"Failed to load model configuration for {args.llm_model}: {err_patch}")

    # Setup BitsAndBytes Quantization Config with fp32 CPU offload enabled
    quantization_config = None
    quant_mode = getattr(args, 'quantization', None)
    if quant_mode in ['4bit', '8bit'] and torch.cuda.is_available():
        if quant_mode == '4bit':
            print("Configuring 4-bit (NF4) double-quantization with bfloat16 compute dtype (fp32 CPU offload enabled)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif quant_mode == '8bit':
            print("Configuring 8-bit quantization (fp32 CPU offload enabled)...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

    attn_impl = getattr(args, 'attn_implementation', 'sdpa')
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        attn_impl = "eager"

    # Support both newer (v4.44+ 'dtype') and older ('torch_dtype') transformers APIs to prevent deprecation warnings
    dtype_kwargs = {}
    try:
        import inspect
        if 'dtype' in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
            dtype_kwargs['dtype'] = torch.bfloat16
        else:
            dtype_kwargs['torch_dtype'] = torch.bfloat16
    except Exception:
        dtype_kwargs['torch_dtype'] = torch.bfloat16

    if torch.cuda.is_available():
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            config=config,
            quantization_config=quantization_config,
            token=auth_token,
            trust_remote_code=trust_remote,
            attn_implementation=attn_impl,
            device_map="auto",
            **dtype_kwargs
        )
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            config=config,
            torch_dtype=torch.float32,
            token=auth_token,
            trust_remote_code=trust_remote
        ).to(device_llm)

    # Safe pad token handling for quantized models (NEVER call resize_token_embeddings!)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.bos_token
    if hasattr(llm_model, 'config'):
        llm_model.config.pad_token_id = tokenizer.pad_token_id

    # Dynamically resolve multi-token EOS stop sequences for modern LLMs
    llm_eos_ids = resolve_eos_ids(tokenizer, llm_model)
    args.eos_token_ids = llm_eos_ids
    print(f"Resolved EOS token IDs for {args.llm_model}: {llm_eos_ids}")

    # Now load DeBERTa Model in standard 32-bit precision on GPU (standard precision)
    target_deberta_dev = getattr(args, 'deberta_device', 'cuda')
    device_deberta = target_deberta_dev if (target_deberta_dev != 'cuda' or torch.cuda.is_available()) else 'cpu'
    print(f"Loading DeBERTa Model on {device_deberta}: microsoft/{args.deberta_model}...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{args.deberta_model}", token=auth_token)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{args.deberta_model}",
        token=auth_token
    ).to(device_deberta)

    # 4. Load BLEURT Metric (Permanent bleurt-base-128 standard)
    print("Loading BLEURT metric for gold standard evaluation (CPU)...")
    try:
        bleurt = evaluate.load("bleurt", "bleurt-base-128")
    except Exception:
        try:
            bleurt = evaluate.load("bleurt")
        except Exception:
            print("Warning: BLEURT metric could not be loaded. BLEURT scores will default to string matching.")
            bleurt = None

    print(f"\nProcessing {len(dataset_subset)} samples (indices {args.start_idx} to {end_idx - 1})...")

    err_count = 0

    # 5. Main Inference & Uncertainty Loop
    for sample in tqdm(dataset_subset, desc=f"Running ICE Quantized ({args.run_id})"):
        try:
            sample_idx = int(sample['question_id'])

            # Fast resume: check if valid completed file already exists
            if not getattr(args, 'force_overwrite', False):
                outfile_path = os.path.join(base_path, f'results_dict_{sample_idx}.pkl.gz')
                alt_outfile_path = os.path.join(base_path, f'results_dict_{sample_idx}.pkl')
                if os.path.exists(outfile_path) or os.path.exists(alt_outfile_path):
                    target_f = outfile_path if os.path.exists(outfile_path) else alt_outfile_path
                    try:
                        if target_f.endswith('.gz'):
                            with gzip.open(target_f, 'rb') as f:
                                existing_data = pickle.load(f)
                        else:
                            with open(target_f, 'rb') as f:
                                existing_data = pickle.load(f)
                        if isinstance(existing_data, dict) and 'ice' in existing_data:
                            continue
                    except Exception:
                        try:
                            os.remove(target_f)
                        except Exception:
                            pass

            # Extract sample fields
            sample_data = extract_sample_data(sample, args.dataset)
            question = sample_data['question']
            choices = sample_data['choices']
            reference_answers = sample_data['reference_answers']

            use_options = getattr(args, 'use_options', False)
            effective_choices = choices if use_options else None

            # Direct Most Likely Generation on 70B model
            direct_prompt = format_answer_prompt(
                tokenizer=tokenizer,
                question=question,
                choices=effective_choices,
                dataset_name=args.dataset
            )
            direct_gen = generate_sequence(
                model=llm_model,
                tokenizer=tokenizer,
                prompt_str=direct_prompt,
                do_sample=args.do_sample_most_likely,
                temperature=args.temperature_most_likely,
                top_p=args.top_p_most_likely,
                num_beams=args.num_beams_most_likely,
                max_new_tokens=args.max_length_of_generated_sequence,
                device=device_llm,
                eos_token_ids=llm_eos_ids
            )

            # Compute BLEURT score vs reference answers (Benchmark invariance: standard threshold = 0.0)
            if bleurt is not None and len(reference_answers) > 0 and reference_answers != [""]:
                bleurt_res = bleurt.compute(
                    predictions=[direct_gen['generation_text']] * len(reference_answers),
                    references=reference_answers
                )
                bleurt_score = float(max(bleurt_res['scores']))
            else:
                bleurt_score = 1.0 if any(str(ans).lower() in direct_gen['generation_text'].lower() for ans in reference_answers if ans) else -1.0

            # Retrieve clarifications directly from cache
            sample_key = str(sample_idx)
            if sample_key in clarification_cache:
                clarifications = clarification_cache[sample_key][:num_clarifications]
            else:
                clarifications = [question]

            # Stage 2: Generate responses & compute NLLs on 70B model for each clarification
            gen_texts = []
            nlls = []

            for clar_q in clarifications:
                clar_ans_prompt = format_answer_prompt(
                    tokenizer=tokenizer,
                    question=clar_q,
                    choices=effective_choices,
                    dataset_name=args.dataset
                )
                ans_gen = generate_sequence(
                    model=llm_model,
                    tokenizer=tokenizer,
                    prompt_str=clar_ans_prompt,
                    do_sample=args.do_sample_most_likely,
                    temperature=args.temperature_most_likely,
                    top_p=args.top_p_most_likely,
                    num_beams=args.num_beams_most_likely,
                    max_new_tokens=args.max_length_of_generated_sequence,
                    device=device_llm,
                    eos_token_ids=llm_eos_ids
                )

                nll = compute_nll(
                    model=llm_model,
                    prompt_ids=ans_gen['prompt_ids'],
                    gen_ids=ans_gen['generation_ids'],
                    device=device_llm
                )

                gen_texts.append(ans_gen['generation_text'])
                nlls.append(nll)

            # Stage 3: Extract exact ICE uncertainties (Aleatoric & Epistemic)
            ice_uncertainties = compute_ice_uncertainties(
                gen_texts=gen_texts,
                nlls=nlls,
                deberta_model=deberta_model,
                deberta_tokenizer=deberta_tokenizer,
                device_deberta=device_deberta
            )

            # Real-Time Structured Console Logging (Mandated by workspace rules)
            tqdm.write("\n" + "-" * 70)
            tqdm.write(f"[Sample #{sample_idx}] Question: {question[:90]}{'...' if len(question) > 90 else ''}")
            tqdm.write(f"  Direct Response   : {direct_gen['generation_text']}")
            tqdm.write(f"  Gold References   : {reference_answers[:3]}")
            tqdm.write(f"  BLEURT Score      : {bleurt_score:.4f} (Correct: {bleurt_score >= 0.0})")
            for idx_r, (c_q, resp_str) in enumerate(zip(clarifications, gen_texts), 1):
                tqdm.write(f"  Clarification #{idx_r} : {c_q[:50]}... -> {resp_str}")
            tqdm.write(
                f"  ICE Diagnostics   : Epistemic={ice_uncertainties['epistemic']:.4f} | "
                f"Total={ice_uncertainties['total']:.4f} | "
                f"Aleatoric={ice_uncertainties['aleatoric']:.4f} | "
                f"Clusters={ice_uncertainties['num_clusters']}"
            )
            tqdm.write("-" * 70)

            # Store result dictionary
            results_dict = {
                'question_id': sample_idx,
                'question': question,
                'choices': choices,
                'effective_choices': effective_choices,
                'reference_answers': reference_answers,
                'most_likely_generation': direct_gen,
                'correctness_dict': {'bleurt': bleurt_score},
                'ice_clarifications': clarifications,
                'ice': {
                    'responses': gen_texts,
                    'nlls': nlls,
                    'epistemic_uncertainty': ice_uncertainties['epistemic'],
                    'total_uncertainty': ice_uncertainties['total'],
                    'aleatoric_uncertainty': ice_uncertainties['aleatoric'],
                    'semantic_entropy_norm': ice_uncertainties['semantic_entropy_norm'],
                    'num_clusters': ice_uncertainties['num_clusters']
                }
            }

            # Save gzip pickle
            outfile_path = os.path.join(base_path, f'results_dict_{sample_idx}.pkl.gz')
            with gzip.open(outfile_path, 'wb') as outfile:
                pickle.dump(results_dict, outfile)

        except KeyboardInterrupt:
            print("\n[Interrupted] Run halted by user (KeyboardInterrupt). Finalizing partial progress...")
            break
        except Exception as err:
            err_count += 1
            print(f"Error processing sample {sample_idx if 'sample_idx' in locals() else 'unknown'}: {err}")
            traceback.print_exc()

    print(f"\nICE quantized experiment completed. Output saved to {base_path}. Total errors: {err_count}")

    # Automatically aggregate all sample outputs into final summary pickles & CSV for downstream analysis
    try:
        from aggregate_results import aggregate_ice_results
        print(f"\n[Auto-Aggregation] Generating final consolidated summary pickle & CSV for downstream analysis...")
        aggregate_ice_results(target_run_id=args.run_id)
    except Exception as agg_err:
        print(f"Note: Automatic result aggregation skipped or encountered: {agg_err}")
        print(f"You can run 'python aggregate_results.py {args.run_id}' to generate the consolidated summary file.")


if __name__ == '__main__':
    run_ice_quantized_experiment()
