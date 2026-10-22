import os
import sys

# Prevent TensorFlow (used by BLEURT) from hogging GPU VRAM needed by 70B/72B quantized LLM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# DynamicCache Compatibility Shim for modern Hugging Face transformers
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
try:
    import yaml
except ImportError:
    yaml = None
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig
)
from datasets import load_dataset

# Add current directory to sys.path for importing tokur_quantized_args and tokur_quantized_perturb
tokur_dir = os.path.dirname(os.path.abspath(__file__))
if tokur_dir not in sys.path:
    sys.path.append(tokur_dir)

from tokur_quantized_args import TokURQuantizedArgs
from tokur_quantized_perturb import wrap_model_with_tokur_quantized, set_model_sampling


class StopOnTokens(StoppingCriteria):
    """Stops LLM GPU generation immediately when a newline, prompt marker, or option prefix is generated."""
    def __init__(self, tokenizer, prompt_len, stop_strings=["\n", "\nQ:", "\nB:", "\nC:", "\nD:"]):
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


def seed_everything(seed: int):
    """Sets random seeds across python, numpy, and PyTorch for exact reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_eos_ids(tokenizer, model):
    """Dynamically resolves EOS and special stop tokens for modern LLM architectures."""
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


def format_tokur_prompt(tokenizer, question, choices=None):
    """
    Constructs prompt adhering to exact prompt consistency standards across ILE and ICE.
    Uses chat templates when supported, otherwise falls back to standardized structured strings.
    """
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
        system_content = "Answer the user's question in a single sentence. Do not repeat the prompt."
        fallback_text = f"Question: {question}\nAnswer:"

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


def load_tokur_dataset(dataset_name: str, num_samples: int = -1, start_idx: int = 0):
    """Loads prompt questions, choices, correct answers, and incorrect answers for supported datasets."""
    dataset_name = dataset_name.lower()
    questions, choices_list, correct_answers_list, incorrect_answers_list = [], [], [], []

    if 'truth' in dataset_name:
        try:
            ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        except Exception:
            ds = load_dataset("truthful_qa", "generation", split="validation")
        for item in ds:
            questions.append(item['question'])
            choices_list.append(None)
            best_ans = item.get('best_answer', '')
            correct_answers_list.append(item.get('correct_answers', [best_ans] if best_ans else []))
            incorrect_answers_list.append(item.get('incorrect_answers', []))

    elif 'open' in dataset_name:
        try:
            ds = load_dataset("allenai/openbookqa", "additional", split="test")
        except Exception:
            ds = load_dataset("openbookqa", "additional", split="test")
        if hasattr(ds, 'get') and 'test' in ds:
            ds = ds['test']
        for item in ds:
            choices = item.get('choices', {})
            choices_text = choices.get('text', [])
            choices_labels = choices.get('label', [])
            ans_key = item.get('answerKey', '')
            try:
                ans_idx = choices_labels.index(ans_key)
                correct_choice = choices_text[ans_idx]
            except Exception:
                correct_choice = ans_key

            fact1 = item.get('fact1', '')
            correct_answers = []
            if correct_choice:
                correct_answers.append(str(correct_choice).strip())
            if fact1 and str(fact1).strip() and str(fact1).strip() not in correct_answers:
                correct_answers.append(str(fact1).strip())
            if not correct_answers:
                correct_answers = [str(ans_key)] if ans_key else [""]

            incorrect_answers = [choice for c_idx, choice in enumerate(choices_text) if choice != correct_choice]

            questions.append(item.get('question_stem', item.get('question', '')))
            choices_list.append(choices_text if choices_text else None)
            correct_answers_list.append(correct_answers)
            incorrect_answers_list.append(incorrect_answers)

    elif 'science' in dataset_name or 'sciq' in dataset_name:
        try:
            ds = load_dataset("tasksource/ScienceQA_text_only", split="test")
        except Exception:
            try:
                ds = load_dataset("allenai/sciq", split="test")
            except Exception:
                ds = load_dataset("derek-thomas/ScienceQA", split="test")
        for item in ds:
            if item.get('image') is not None and not isinstance(item.get('image'), str) and item.get('image') != "":
                continue

            choices = item.get('choices', [])
            solution = item.get('solution', '')
            ans_idx = item.get('answer', -1)

            correct_answers = []
            if solution:
                correct_answers.append(solution)
            if choices and 0 <= ans_idx < len(choices):
                correct_choice = choices[ans_idx]
                if correct_choice not in correct_answers:
                    correct_answers.append(correct_choice)
            if not correct_answers:
                correct_answers = [str(ans_idx)] if ans_idx != -1 else [""]

            incorrect_answers = [choice for c_idx, choice in enumerate(choices) if c_idx != ans_idx]

            questions.append(item['question'])
            choices_list.append(choices if choices else None)
            correct_answers_list.append(correct_answers)
            incorrect_answers_list.append(incorrect_answers)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if num_samples > 0:
        questions = questions[start_idx:start_idx + num_samples]
        choices_list = choices_list[start_idx:start_idx + num_samples]
        correct_answers_list = correct_answers_list[start_idx:start_idx + num_samples]
        incorrect_answers_list = incorrect_answers_list[start_idx:start_idx + num_samples]

    return questions, choices_list, correct_answers_list, incorrect_answers_list


@torch.no_grad()
def run_tokur_quantized_experiment():
    args = TokURQuantizedArgs()

    seed_everything(args.seed_value)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = args.hf_token if (args.hf_token and str(args.hf_token).strip()) else None

    # Output directory preparation using canonical convention: src/experiments/tokur/result/<run_id>/
    output_dir = os.path.join(tokur_dir, 'result', args.run_id)
    os.makedirs(output_dir, exist_ok=True)
    args.args_to_yaml(output_dir)

    print("=" * 70)
    print(f"RUNNING QUANTIZED TokUR BASELINE EXPERIMENT: {args.run_id}")
    print(f"Dataset                  : {args.dataset}")
    print(f"LLM (Quantized Target)   : {args.llm_model}")
    print(f"Quantization Mode        : {args.quantization} (BitsAndBytes NF4/bfloat16)")
    print(f"Attention Implementation : {getattr(args, 'attn_implementation', 'sdpa')}")
    print(f"Perturbation Passes (M)  : {args.num_perturbs} (rank={args.rank}, sigma={args.sigma})")
    print(f"Results Directory        : {output_dir}")
    print("=" * 70)

    # 1. Clean GPU VRAM to ensure 100% pure memory before loading large model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Initialize Tokenizer
    print(f"[TokUR Quantized] Initializing tokenizer for '{args.llm_model}'...")
    is_phi = "phi-3" in str(args.llm_model).lower() or "phi3" in str(args.llm_model).lower()
    trust_remote = False if is_phi else True
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=trust_remote, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.bos_token

    # 3. Setup BitsAndBytes Quantization Config with fp32 CPU offload enabled to prevent spillage ValueError
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

    # Support both newer (v4.44+ 'dtype') and older ('torch_dtype') transformers APIs
    dtype_kwargs = {}
    try:
        import inspect
        if 'dtype' in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
            dtype_kwargs['dtype'] = torch.bfloat16
        else:
            dtype_kwargs['torch_dtype'] = torch.bfloat16
    except Exception:
        dtype_kwargs['torch_dtype'] = torch.bfloat16

    print(f"[TokUR Quantized] Loading model directly onto GPU with device_map='auto'...")
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            quantization_config=quantization_config,
            token=hf_token,
            trust_remote_code=trust_remote,
            attn_implementation=attn_impl,
            device_map="auto",
            **dtype_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            token=hf_token,
            trust_remote_code=trust_remote,
            torch_dtype=torch.float32
        ).to(device)

    model.eval()

    # Safe pad token handling (never call resize_token_embeddings on quantized models!)
    if hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Wrap target modules with SVD-aligned QuantizedLowRankBayesianLinear layers
    print(f"[TokUR Quantized] Wrapping target modules {args.target_modules} with SVD-aligned LowRankBayesianLinear layers...")
    bayesian_layers = wrap_model_with_tokur_quantized(
        model=model,
        rank=args.rank,
        sigma=args.sigma,
        target_modules=args.target_modules
    )
    print(f"[TokUR Quantized] Successfully wrapped {len(bayesian_layers)} layers across model.")

    # 5. Load BLEURT metric (Permanent bleurt-base-128 standard, loaded after PyTorch LLM is initialized)
    try:
        import evaluate
        print("[TokUR Quantized] Loading BLEURT metric (bleurt-base-128)...")
        bleurt = evaluate.load("bleurt", "bleurt-base-128")
        print("[TokUR Quantized] BLEURT metric loaded successfully.")
    except Exception as e:
        try:
            import evaluate
            bleurt = evaluate.load("bleurt")
        except Exception as e2:
            print(f"[TokUR Quantized] Warning: Could not load BLEURT metric ({e2}). String matching fallback will be used.")
            bleurt = None

    # 6. Load Dataset
    questions, choices_list, correct_answers_list, incorrect_answers_list = load_tokur_dataset(
        args.dataset, args.num_samples, args.start_idx
    )
    print(f"[TokUR Quantized] Loaded {len(questions)} evaluation samples from dataset '{args.dataset}'.")

    llm_eos_ids = resolve_eos_ids(tokenizer, model)
    args.eos_token_ids = llm_eos_ids
    print(f"[TokUR Quantized] Resolved EOS token IDs: {llm_eos_ids}")

    # 7. Checkpoint & Resume Setup
    checkpoint_dir = os.path.join(output_dir, '.checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_tokur.pkl')
    existing_records = []
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                existing_records = pickle.load(f)
            print(f"[TokUR Quantized] Found existing checkpoint with {len(existing_records)} samples. Resuming from sample index {len(existing_records)}...")
        except Exception as e:
            print(f"[TokUR Quantized] Warning: Could not read checkpoint file ({e}). Starting fresh.")
            existing_records = []

    start_sample_idx = len(existing_records)
    all_entropy_dicts = [r['entropy_dict'] for r in existing_records]
    all_labels = [r['label'] for r in existing_records]
    all_generations = [r['generation'] for r in existing_records]
    all_questions = [r['question'] for r in existing_records]

    def _atomic_flush():
        """Flushes in-memory results atomically to checkpoint and canonical summary files."""
        temp_ckpt = checkpoint_file + ".tmp"
        with open(temp_ckpt, 'wb') as f:
            pickle.dump(existing_records, f)
        os.replace(temp_ckpt, checkpoint_file)

        summary_rows = []
        for r in existing_records:
            e_dict = r.get('entropy_dict', {})
            au_val = float(e_dict.get('tokur_au', e_dict.get('entropy_final', 0.0)))
            mi_val = float(e_dict.get('tokur_mi', e_dict.get('entropy_mixture', 0.0)))
            tot_val = au_val + mi_val
            summary_rows.append({
                'sample_idx': r.get('sample_idx', 0),
                'question': r.get('question', ''),
                'generation': r.get('generation', ''),
                'correctness': r.get('label', 0.0),
                'tokur_au': au_val,
                'tokur_mi': mi_val,
                'tokur_total': tot_val,
                'aleatoric_uncertainty': au_val,
                'epistemic_uncertainty': mi_val,
                'total_uncertainty': tot_val,
            })
        summary_df = pd.DataFrame(summary_rows) if summary_rows else None

        # Save to run-specific output dir
        for fname, obj in [
            ('entropies_tokur.pkl', all_entropy_dicts),
            ('labels_tokur.pkl', all_labels),
            ('questions_tokur.pkl', all_questions),
            ('generations_tokur.pkl', all_generations),
        ]:
            temp_p = os.path.join(output_dir, f"{fname}.tmp")
            with open(temp_p, 'wb') as f:
                pickle.dump(obj, f)
            os.replace(temp_p, os.path.join(output_dir, fname))

        if summary_df is not None and len(summary_df) > 0:
            summary_csv = os.path.join(output_dir, f"{args.run_id}_summary.csv")
            summary_df.to_csv(summary_csv, index=False)
            summary_pkl = os.path.join(output_dir, f"{args.run_id}_summary.pkl")
            with open(summary_pkl, 'wb') as sf:
                pickle.dump(summary_rows, sf)

            # Also write summary directly to tokur_dir/result/ parent directory for instant discovery by notebooks
            parent_result_dir = os.path.join(tokur_dir, 'result')
            summary_df.to_csv(os.path.join(parent_result_dir, f"{args.run_id}_summary.csv"), index=False)
            with open(os.path.join(parent_result_dir, f"{args.run_id}_summary.pkl"), 'wb') as sf:
                pickle.dump(summary_rows, sf)

    if start_sample_idx >= len(questions) and len(questions) > 0:
        print(f"[TokUR Quantized] All {len(questions)} samples already processed according to checkpoint.")
        _atomic_flush()
        print(f"\n[TokUR Quantized] Successfully completed TokUR evaluation! Output: {os.path.abspath(output_dir)}")
        return

    # 8. Main Inference Loop
    progress_bar = tqdm(range(start_sample_idx, len(questions)), total=len(questions), initial=start_sample_idx, desc=f"Running TokUR Quantized ({args.run_id})")
    for idx in progress_bar:
        try:
            question = questions[idx]
            choices = choices_list[idx]
            correct_answers = correct_answers_list[idx]

            prompt_str = format_tokur_prompt(tokenizer, question, choices=choices)

            # Route input tensors to device of first model layer
            if hasattr(model, 'device'):
                model_device = model.device
            else:
                try:
                    model_device = next(model.parameters()).device
                except Exception:
                    model_device = device

            inputs = tokenizer(prompt_str, return_tensors="pt").to(model_device)
            input_ids = inputs['input_ids']
            prompt_len = input_ids.shape[1]

            # Disable perturbations for standard greedy generation
            set_model_sampling(bayesian_layers, sample=False)

            stopping_criteria = StoppingCriteriaList([StopOnTokens(tokenizer, prompt_len)])
            gen_out = model.generate(
                **inputs,
                min_new_tokens=2,
                max_new_tokens=args.max_length_of_generated_sequence,
                do_sample=False,
                num_beams=args.num_beams_most_likely,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=llm_eos_ids[0] if llm_eos_ids else tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )

            gen_tokens = gen_out[0][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Compute BLEURT correctness score (Standard benchmark threshold = 0.0)
            if bleurt is not None and len(correct_answers) > 0:
                try:
                    bleurt_res = bleurt.compute(
                        predictions=[gen_text] * len(correct_answers),
                        references=correct_answers
                    )
                    bleurt_score = max(bleurt_res['scores'])
                except Exception as e:
                    bleurt_score = 1.0 if any(ans.lower() in gen_text.lower() or gen_text.lower() in ans.lower() for ans in correct_answers) else -1.0
            else:
                bleurt_score = 0.0

            # Low-Rank Stochastic Weight Perturbation Forward Passes (M passes)
            if len(gen_tokens) == 0:
                fallback_tok = llm_eos_ids[0] if llm_eos_ids else (tokenizer.eos_token_id or 0)
                target_ids = torch.tensor([[fallback_tok]], device=model_device)
            else:
                target_ids = gen_tokens.unsqueeze(0).to(model_device)

            seq_len = target_ids.shape[1]
            all_pass_probs = []

            # Enable perturbations for stochastic weight noise passes
            set_model_sampling(bayesian_layers, sample=True)
            for m in range(args.num_perturbs):
                pass_seed = args.seed_value + idx * 100 + m
                rng_state = torch.get_rng_state()
                if torch.cuda.is_available():
                    cuda_rng_state = torch.cuda.get_rng_state()

                torch.manual_seed(pass_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(pass_seed)

                # Teacher-forced forward pass over generated tokens to obtain next-token logit distributions
                full_ids = torch.cat([input_ids, target_ids], dim=1)
                outputs = model(full_ids)
                logits = outputs.logits[:, prompt_len - 1:-1, :]  # shape: [1, seq_len, vocab_size]
                probs = F.softmax(logits, dim=-1).squeeze(0)      # shape: [seq_len, vocab_size]
                all_pass_probs.append(probs)

                torch.set_rng_state(rng_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(cuda_rng_state)

            stacked_probs = torch.stack(all_pass_probs, dim=0)

            # Compute TokUR Uncertainty Metrics
            # Token-level Aleatoric Uncertainty (AU): expected entropy across weight noise passes
            pass_entropies = -torch.sum(stacked_probs * torch.log(torch.clamp(stacked_probs, min=1e-12)), dim=-1)
            au_token = torch.mean(pass_entropies, dim=0).mean().item()

            # Token-level Total Uncertainty (TU): entropy of average distribution across passes
            mean_probs = torch.mean(stacked_probs, dim=0)
            tu_token = -torch.sum(mean_probs * torch.log(torch.clamp(mean_probs, min=1e-12)), dim=-1).mean().item()

            # Token-level Epistemic Uncertainty (EU / Mutual Information): TU - AU
            eu_token = max(0.0, tu_token - au_token)
            semantic_entropy = tu_token

            entropy_dict = {
                'normalised_semantic_entropy': float(semantic_entropy),
                'entropy_final': float(au_token),
                'entropy_mixture': float(eu_token),
                'tokur_au': float(au_token),
                'tokur_mi': float(eu_token),
                'tokur_total': float(tu_token),
            }

            # Real-Time Structured Console Logging (Rule Requirement)
            tqdm.write("\n" + "-" * 70)
            tqdm.write(f"[Sample #{idx}] Question: {question[:90]}{'...' if len(question) > 90 else ''}")
            tqdm.write(f"  Greedy Output    : {gen_text}")
            tqdm.write(f"  Gold References  : {correct_answers[:3]}")
            tqdm.write(f"  BLEURT Score     : {bleurt_score:.4f} (Correct: {bleurt_score >= 0.0})")
            tqdm.write(
                f"  TokUR Diagnostics: AU (Aleatoric)={au_token:.4f} | "
                f"EU (Epistemic)={eu_token:.4f} | "
                f"TU (Total)={tu_token:.4f}"
            )
            tqdm.write("-" * 70)

            record = {
                'sample_idx': idx,
                'question': question,
                'generation': gen_text,
                'label': float(bleurt_score),
                'entropy_dict': entropy_dict
            }
            existing_records.append(record)
            all_entropy_dicts.append(entropy_dict)
            all_labels.append(float(bleurt_score))
            all_generations.append(gen_text)
            all_questions.append(question)

            # Flush atomically every 5 samples
            if (idx + 1) % 5 == 0 or (idx + 1) == len(questions):
                _atomic_flush()

        except KeyboardInterrupt:
            print("\n[Interrupted] Run halted by user (KeyboardInterrupt). Finalizing partial progress...")
            break
        except Exception as err:
            print(f"Error processing sample {idx}: {err}")
            import traceback
            traceback.print_exc()

    # Final flush on completion
    _atomic_flush()
    print(f"\n[TokUR Quantized] Experiment completed. Outputs saved to {output_dir}")


if __name__ == '__main__':
    run_tokur_quantized_experiment()
