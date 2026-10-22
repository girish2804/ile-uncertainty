import os
import sys

# Prevent TensorFlow (used by BLEURT) from hogging the entire GPU VRAM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

# DynamicCache Compatibility Shim for Phi-3 / DeepSeek MoE ('seen_tokens', 'get_max_length' errors)
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

import math
import random
import argparse
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

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset
from torch.utils.data import DataLoader

# Add current directory to sys.path for importing ice_args
ice_dir = os.path.dirname(os.path.abspath(__file__))
if ice_dir not in sys.path:
    sys.path.append(ice_dir)

from ice_args import ICEArgs


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
        # Require at least 2 tokens generated so leading newlines from chat templates don't trigger immediate stop
        if len(gen_ids) <= 1:
            return False
        latest_ids = gen_ids[-16:]
        decoded_text = self.tokenizer.decode(latest_ids, skip_special_tokens=True)
        return any(stop in decoded_text for stop in self.stop_strings)


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


@torch.no_grad()
def generate_sequence(model, tokenizer, prompt_str, do_sample, temperature, top_p, num_beams, max_new_tokens, device, stop_on_question=False, eos_token_ids=None):
    """
    Direct HuggingFace generation. Retains original logic for standard models, with isolated Gemma-specific handling.
    """
    model_type_str = str(getattr(model.config, 'model_type', '')).lower()
    is_gemma_model = "gemma" in model_type_str or "gemma" in str(type(model)).lower()

    if is_gemma_model:
        # Isolated generation logic specifically for Gemma 2 / Gemma 3 models
        formatted_prompt = prompt_str
        if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None) is not None:
            try:
                clean_user_input = prompt_str.replace("Q:", "").replace("A:", "").strip()
                messages = [{"role": "user", "content": clean_user_input}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = prompt_str

        inputs = tokenizer(formatted_prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        prompt_len = input_ids.shape[1]
        
        pad_id = tokenizer.pad_token_id
        if pad_id is None or pad_id == tokenizer.eos_token_id:
            pad_id = tokenizer.convert_tokens_to_ids("<pad>")
            if pad_id is None or pad_id == tokenizer.eos_token_id or pad_id == tokenizer.unk_token_id:
                pad_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

        if eos_token_ids is not None and len(eos_token_ids) > 0:
            eos_id = eos_token_ids
        else:
            eos_id = getattr(model.config, 'eos_token_id', tokenizer.eos_token_id)
            if isinstance(eos_id, list):
                eos_id = eos_id[0]

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "min_new_tokens": 2,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id
        }

        if stop_on_question:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(tokenizer, prompt_len=prompt_len, stop_strings=["?", "\n", "Original Question"])])

        if do_sample:
            gen_kwargs.update({"do_sample": True, "temperature": max(temperature, 0.01), "top_p": top_p})
        else:
            gen_kwargs.update({"do_sample": False, "num_beams": max(num_beams, 1)})

        output_ids = model.generate(**gen_kwargs)[0]
        gen_ids = output_ids[prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Clean text without destroying outputs that start with Q: or A:
        first_line = gen_text.split('\n')[0].strip()
        clean_text = first_line.split('Original Question')[0].split('Clarified Question')[0].strip()
        if clean_text.startswith('A:'):
            clean_text = clean_text[2:].strip()
        elif clean_text.startswith('Q:'):
            clean_text = clean_text[2:].strip()

        if set(clean_text).issubset({'.', ' ', ':', '-'}) or len(clean_text) == 0:
            clean_text = gen_text.strip()

        if len(clean_text) > 0:
            gen_text = clean_text
            gen_ids = tokenizer.encode(gen_text, add_special_tokens=False, return_tensors='pt')[0]

        return {
            'prompt_ids': input_ids[0].cpu(),
            'generation_ids': gen_ids.cpu(),
            'generation_text': gen_text
        }

    # --- Original General Generation Logic (Conserved for all non-Gemma models) ---
    input_ids = tokenizer.encode(prompt_str, return_tensors='pt').to(device)
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
            "temperature": temperature,
            "top_p": top_p
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": num_beams
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
    """
    if len(gen_ids) == 0:
        return 0.0

    full_ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(device)
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
    tmp_filepath = target_filepath + ".tmp"
    with open(tmp_filepath, 'wb') as f:
        pickle.dump(cache_dict, f)
    os.replace(tmp_filepath, target_filepath)


def format_answer_prompt(tokenizer, question, choices=None, dataset_name='scienceqa'):
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


def extract_sample_data(sample, dataset_name='scienceqa'):
    """
    Extracts question, choices, and correct reference answers consistently across datasets.
    """
    clean_d = dataset_name.lower().replace('_', '').replace('-', '')

    if 'science' in clean_d or 'sciq' in clean_d:
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

        # Reference answer logic (exact ILE run_experiment_openbookqa.py strategy):
        # Prioritize correct_choice as primary reference, include fact1 as second reference
        correct_answers = []
        if correct_choice:
            correct_answers.append(correct_choice)
        if fact1 and str(fact1).strip() and str(fact1).strip() not in correct_answers:
            correct_answers.append(str(fact1).strip())
        if not correct_answers:
            correct_answers = [str(ans_key)] if ans_key else [""]

        # Use unlettered choices_text matching ILE run_experiment_openbookqa.py and ScienceQA
        return {
            'question': question,
            'choices': choices_text,
            'reference_answers': correct_answers
        }

    else:  # TruthfulQA / default
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


def run_ice_experiment():
    args = ICEArgs()
    
    num_clarifications = getattr(args, 'num_clarifications', 5)

    model_name_clean = args.llm_model.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "")
    dataset_clean = getattr(args, 'dataset', 'truthfulqa').lower()
    args.run_id = f"ice-{dataset_clean}-{model_name_clean}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'result', args.run_id)
    os.makedirs(base_path, exist_ok=True)

    # Save run arguments
    args.args_to_yaml(base_path)

    print("=" * 60)
    print(f"RUNNING STRICT ICE BASELINE EXPERIMENT: {args.run_id}")
    print(f"Dataset: {args.dataset}")
    print(f"LLM: {args.llm_model}")
    print(f"Clarifications per sample: {num_clarifications}")
    print(f"Results Directory: {base_path}")
    print("=" * 60)

    seed_everything(seed=args.seed_value)

    device_llm = "cuda" if torch.cuda.is_available() else "cpu"
    device_deberta = "cuda" if torch.cuda.is_available() else "cpu"

    auth_token = getattr(args, 'hf_token', None) or os.environ.get("HF_TOKEN", os.environ.get('HF_TOKEN', None))

    # Load DeBERTa model for semantic clustering
    print(f"Loading DeBERTa Model: microsoft/{args.deberta_model}...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{args.deberta_model}")
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{args.deberta_model}"
    ).to(device_deberta)

    # Load LLM
    print(f"Loading LLM Model: {args.llm_model}...")
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
                # Fix LLaMA 3.2 rope_scaling format for older transformers library versions
                if "rope_type" in rs or "high_freq_factor" in rs or len(rs) > 2:
                    config_dict["rope_scaling"] = {
                        "type": "linear",
                        "factor": float(rs.get("factor", 32.0))
                    }
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            model_type = config_dict.get("model_type", "llama")
            
            # Fallback for newer model types not registered in older transformers versions
            if model_type not in CONFIG_MAPPING:
                if model_type.startswith("gemma") and "gemma" in CONFIG_MAPPING:
                    print(f"Fallback: mapping unrecognized '{model_type}' to 'gemma' architecture...")
                    model_type = "gemma"
                    config_dict["model_type"] = "gemma"
                elif model_type.startswith("qwen") and "qwen2" in CONFIG_MAPPING:
                    model_type = "qwen2"
                    config_dict["model_type"] = "qwen2"
                elif model_type.startswith("phi") and "phi" in CONFIG_MAPPING:
                    model_type = "phi"
                    config_dict["model_type"] = "phi"

            if model_type in CONFIG_MAPPING:
                config = CONFIG_MAPPING[model_type].from_dict(config_dict)
            else:
                config_dict.pop("model_type", None)
                config = AutoConfig.for_model(model_type, **config_dict)
        except Exception as err_patch:
            raise RuntimeError(f"Failed to load model configuration for {args.llm_model}: {err_patch}")

    quantization_config = None
    quant_mode = getattr(args, 'quantization', 'none')
    if quant_mode in ['4bit', '8bit'] and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        if quant_mode == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quant_mode == '8bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if torch.cuda.is_available():
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            token=auth_token,
            trust_remote_code=trust_remote,
            device_map="auto"
        )
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            config=config,
            torch_dtype=torch.float32,
            token=auth_token,
            trust_remote_code=trust_remote
        ).to(device_llm)

    if tokenizer.pad_token is None:
        if "<pad>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<pad>"
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            llm_model.resize_token_embeddings(len(tokenizer))
    llm_model.config.pad_token_id = tokenizer.pad_token_id

    # Dynamically resolve multi-token EOS stop sequences for modern LLMs (Qwen2.5, Phi-3, LLaMA-3)
    llm_eos_ids = resolve_eos_ids(tokenizer, llm_model)
    args.eos_token_ids = llm_eos_ids
    print(f"Resolved EOS token IDs for {args.llm_model}: {llm_eos_ids}")

    # Load BLEURT metric for gold standard correctness (Permanent bleurt-base-128 standard)
    print("Loading BLEURT metric for gold standard evaluation...")
    try:
        bleurt = evaluate.load("bleurt", "bleurt-base-128")
    except Exception:
        try:
            bleurt = evaluate.load("bleurt")
        except Exception:
            print("Warning: BLEURT metric could not be loaded. BLEURT scores will default to N/A.")
            bleurt = None

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    clean_dataset = getattr(args, 'dataset', 'scienceqa').lower().replace('_', '').replace('-', '')
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

    # Setup clarification cache directory & file
    cache_dir = os.path.join(script_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    clar_model_name = getattr(args, 'clarification_model', 'meta-llama/Meta-Llama-3-8B-Instruct')
    clar_model_clean = clar_model_name.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "")
    cache_file = os.path.join(cache_dir, f"clarifications_{dataset_clean}_{clar_model_clean}.pkl")

    clarification_cache = {}
    if getattr(args, 'cache_clarifications', True) and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                clarification_cache = pickle.load(f)
            print(f"Loaded existing clarification cache ({len(clarification_cache)} samples) from {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to load clarification cache from {cache_file}: {e}")
            clarification_cache = {}

    print(f"Processing {len(dataset_subset)} samples (indices {args.start_idx} to {end_idx - 1})...")

    err_count = 0
    new_cache_entries = 0

    clar_model = None
    clar_tokenizer = None

    def get_clarification_generator():
        nonlocal clar_model, clar_tokenizer
        if clar_model is None:
            print(f"\n[Lazy Load] Cache miss detected. Loading clarification generator model: {args.clarification_model}...")
            is_clar_phi = "phi-3" in str(args.clarification_model).lower() or "phi3" in str(args.clarification_model).lower()
            clar_trust_remote = False if is_clar_phi else True
            try:
                clar_tokenizer = AutoTokenizer.from_pretrained(args.clarification_model, token=auth_token, trust_remote_code=clar_trust_remote)
            except Exception as err_tok:
                print(f"[Lazy Load] Fast tokenizer loading failed ({err_tok}). Retrying with use_fast=False...")
                clar_tokenizer = AutoTokenizer.from_pretrained(args.clarification_model, token=auth_token, use_fast=False, trust_remote_code=clar_trust_remote)
            if torch.cuda.is_available():
                clar_model = AutoModelForCausalLM.from_pretrained(
                    args.clarification_model,
                    quantization_config=quantization_config, # Reuse same quant config if defined
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
                if clar_tokenizer.eos_token is not None:
                    clar_tokenizer.pad_token = clar_tokenizer.eos_token
                else:
                    clar_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    clar_model.resize_token_embeddings(len(clar_tokenizer))
            clar_model.config.pad_token_id = clar_tokenizer.pad_token_id
            print("[Lazy Load] Clarification model loaded successfully.\n")
        return clar_model, clar_tokenizer
    for sample in tqdm(dataset_subset, desc=f"Running ICE Baseline ({args.run_id})"):
        try:
            sample_idx = int(sample['question_id'])

            # Fast resume: check if valid completed file already exists unless force_overwrite is enabled
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

            # Extract sample fields with strict schema consistency
            sample_data = extract_sample_data(sample, args.dataset)
            question = sample_data['question']
            choices = sample_data['choices']
            reference_answers = sample_data['reference_answers']

            # Automatic options handling: choices are included only for datasets with options
            use_options = getattr(args, 'use_options', any(d in clean_dataset for d in ['science', 'open', 'sciq']))
            effective_choices = choices if use_options else None

            # Direct Most Likely Generation for BLEURT evaluation (consistent prompt)
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

            # Compute BLEURT score vs reference answers
            if bleurt is not None and len(reference_answers) > 0 and reference_answers != [""]:
                bleurt_res = bleurt.compute(
                    predictions=[direct_gen['generation_text']] * len(reference_answers),
                    references=reference_answers
                )
                bleurt_score = float(max(bleurt_res['scores']))
            else:
                bleurt_score = 1.0 if any(str(ans).lower() in direct_gen['generation_text'].lower() for ans in reference_answers if ans) else -1.0

            # Stage 1: Check cache or generate K input clarification questions
            sample_key = str(sample_idx)
            if sample_key in clarification_cache and len(clarification_cache[sample_key]) >= num_clarifications:
                clarifications = clarification_cache[sample_key][:num_clarifications]
            else:
                if getattr(args, 'use_icl_prompt', True):
                    if effective_choices:
                        formatted_choices = ", ".join(effective_choices) if isinstance(effective_choices, list) else str(effective_choices)
                        clarification_prompt = (
                            "Below is a multiple-choice question that may be ambiguous. Please generate a clarified and unambiguous version of the question that still asks for one of the given choices.\n\n"
                            "Choices: France, Argentina, Brazil\n"
                            "Original Question: Who won the world cup?\n"
                            "Clarified Question: Which national team won the FIFA Men's World Cup in 2022?\n\n"
                            "Choices: Pacific Northwest, East Coast, Capital\n"
                            "Original Question: Where is Washington?\n"
                            "Clarified Question: What is the geographic location of Washington State in the US?\n\n"
                            f"Choices: {formatted_choices}\n"
                            f"Original Question: {question}\n"
                            "Clarified Question:"
                        )
                    else:
                        clarification_prompt = (
                            "Below is a potentially ambiguous question. Please generate a clarified and unambiguous version of the question.\n\n"
                            "Original Question: Who won the world cup?\n"
                            "Clarified Question: Which national team won the FIFA Men's World Cup in 2022?\n\n"
                            "Original Question: Where is Washington?\n"
                            "Clarified Question: What is the geographic location of Washington State in the US?\n\n"
                            f"Original Question: {question}\n"
                            "Clarified Question:"
                        )
                else:
                    if effective_choices:
                        formatted_choices = ", ".join(effective_choices) if isinstance(effective_choices, list) else str(effective_choices)
                        clarification_prompt = f"Given choices [{formatted_choices}], a much clearer and unambiguous way to ask the question '{question}' is: "
                    else:
                        clarification_prompt = f"If someone asks the ambiguous question '{question}', a much clearer and unambiguous way to ask it is: "

                clarifications = []
                c_model, c_tokenizer = get_clarification_generator()
                c_eos_ids = resolve_eos_ids(c_tokenizer, c_model)
                for _ in range(num_clarifications):
                    clar_gen = generate_sequence(
                        model=c_model,
                        tokenizer=c_tokenizer,
                        prompt_str=clarification_prompt,
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
                        clarifications.append(clar_gen['generation_text'])

                if len(clarifications) == 0:
                    clarifications = [question]

                # Update clarification cache & save periodically using atomic write
                clarification_cache[sample_key] = clarifications
                new_cache_entries += 1
                if getattr(args, 'cache_clarifications', True) and new_cache_entries % 10 == 0:
                    save_cache_atomically(clarification_cache, cache_file)

            # Stage 2: Generate responses & compute NLLs for each clarification
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

            # Real-Time Structured Console Logging (Mandated by project rules)
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

    if getattr(args, 'cache_clarifications', True) and new_cache_entries > 0:
        save_cache_atomically(clarification_cache, cache_file)
        print(f"Saved final clarification cache ({len(clarification_cache)} total samples) to {cache_file}")

    print(f"ICE baseline experiment completed. Output saved to {base_path}. Total errors: {err_count}")

    # Automatically aggregate all sample outputs into final summary pickles & CSV for downstream notebooks
    try:
        from aggregate_results import aggregate_ice_results
        print(f"\n[Auto-Aggregation] Generating final consolidated summary pickle & CSV for downstream notebooks...")
        try:
            aggregate_ice_results(target_run_id=args.run_id)
        except TypeError:
            try:
                aggregate_ice_results(args.run_id)
            except TypeError:
                aggregate_ice_results()
    except Exception as agg_err:
        print(f"Note: Automatic result aggregation skipped or encountered: {agg_err}")
        print(f"You can run 'python aggregate_results.py {args.run_id}' to generate the consolidated summary file.")


if __name__ == '__main__':
    run_ice_experiment()
