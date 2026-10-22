import os
import sys
import gc
import time
import json
import csv
import numpy as np
import torch
from tqdm import tqdm

# 1. TF & PyTorch CUDA safety flags (strictly before any other imports per project rules)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

# DynamicCache compatibility shim for Phi-3 / Qwen / modern Transformers
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
except Exception:
    pass

# Dynamic path resolution to root, src, experiments, and tokur
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
experiments_dir = os.path.abspath(os.path.join(src_dir, 'experiments'))
tokur_dir = os.path.join(experiments_dir, 'tokur')
for p in [script_dir, src_dir, experiments_dir, tokur_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from disagree import calculate_all_disagreement_metrics
from token_attribution import compute_token_score_ranking, rank_tensor
from layer_ensemble import get_model_layer_count
from tokur_perturb import wrap_model_with_tokur, set_model_sampling, clear_tokur_cache


class BenchmarkCostArgs:
    """
    Dedicated argument configuration class for computational cost benchmarking.
    Configured strictly per project rules: NO CLI parameters, explicit HF token,
    and deterministic OpenBookQA sample evaluation.
    """
    # Class-level attribute fallback (guaranteed on class and all instances)
    alphas = (1/3, 1/3, 1/3)
    relative_top = 0.1
    min_tokens_to_keep = 32
    skip_top_neighbors = 2
    token_prob_threshold = 0.001
    invalid_ids = []
    num_total_generations = 10
    compute_cleaned = False
    seed_value = 42
    eos_token_ids = []

    def __getattr__(self, name):
        if name == 'alphas':
            return (1/3, 1/3, 1/3)
        raise AttributeError(f"'BenchmarkCostArgs' object has no attribute '{name}'")

    def __init__(self):
        self.dataset = 'openbookqa'
        
        # All 9 models evaluated in Table 1 (ordered by family and parameter scale)
        self.models_to_benchmark = [
            'meta-llama/Llama-3.2-1B-Instruct',
            'Qwen/Qwen2.5-1.5B-Instruct',
            'meta-llama/Llama-3.2-3B-Instruct',
            'Qwen/Qwen2.5-3B-Instruct',
            'microsoft/Phi-3-mini-4k-instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct',
            'microsoft/Phi-3-medium-4k-instruct',
            'Qwen/Qwen2.5-14B-Instruct'
        ]
        
        # Run controls:
        # Set to None to sequentially evaluate all 9 models.
        # Or specify a subset list of indices, e.g. [1] for Qwen-1.5B only, or [0, 1].
        self.run_model_indices = None
        self.output_dir_prefix = 'benchmark_cost_efficient'
        self.skip_completed = False     # Set to False to run benchmarks without skipping
        self.force_rerun = True         # Set True to overwrite or re-benchmark models

        self.deberta_model = "microsoft/deberta-large-mnli"
        self.hf_token = os.environ.get('HF_TOKEN', None)
        
        # Benchmarking sample parameters
        self.num_samples = 30   # 30 samples is statistically optimal for GPU latency
        self.num_warmup = 3     # Discard first 3 queries to eliminate JIT/allocation noise
        self.max_new_tokens = 35 # Bounded generation length for single-sentence answer
        
        # Hyperparameters matching Table 1 baselines
        self.num_kuhn_samples = 10      # S = 10 for Kuhn SE (10 sampled sequences + 90 NLI pairs)
        self.num_sdlg_sequences = 10    # S = 10 for SDLG (1 greedy + 9 substitutions + 90 NLI pairs)
        self.num_ice_clarifications = 5 # M = 5 for ICE clarification prompts (5 prompts + 5 responses + 20 NLI pairs)
        self.num_tokur_perturbs = 5     # P = 5 for TokUR low-rank perturbations (5 perturbed forward passes)
        self.tokur_rank = 4
        self.tokur_sigma = 0.05
        
        # Salient layer & SDLG parameters
        self.alphas = (1/3, 1/3, 1/3)   # Attribution, substitution, importance
        self.relative_top = 0.1
        self.min_tokens_to_keep = 32
        self.skip_top_neighbors = 2
        self.token_prob_threshold = 0.001
        self.num_total_generations = 10
        self.compute_cleaned = False
        self.seed_value = 42
        self.eos_token_ids = []
        self.invalid_ids = []


def measure_gpu_execution(fn, *args, **kwargs):
    """
    High-precision GPU wall-clock execution timer and multi-dimensional memory profiler.
    Measures:
      1. Elapsed time using CUDA events (hardware synchronized).
      2. Peak tensor memory allocated (torch.cuda.max_memory_allocated).
      3. Peak physical memory reserved by caching allocator (torch.cuda.max_memory_reserved).
      4. Dynamic memory overhead (Delta VRAM) allocated during inference above static model weights.
    Returns: (output, elapsed_seconds, peak_allocated_gb, peak_reserved_gb, dynamic_overhead_mb)
    """
    if torch.cuda.is_available():
        base_allocated_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output = fn(*args, **kwargs)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_sec = start_event.elapsed_time(end_event) / 1000.0
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        dynamic_overhead_mb = max(0.0, (torch.cuda.max_memory_allocated() - base_allocated_bytes) / (1024 ** 2))
        return output, elapsed_sec, peak_allocated_gb, peak_reserved_gb, dynamic_overhead_mb
    else:
        t0 = time.perf_counter()
        output = fn(*args, **kwargs)
        elapsed_sec = time.perf_counter() - t0
        return output, elapsed_sec, 0.0, 0.0, 0.0


def load_openbookqa_prompts(tokenizer, num_total):
    """
    Loads standardized OpenBookQA test questions formatted identically
    to run_experiment_openbookqa.py to ensure zero prompt format discrepancy.
    """
    try:
        ds = load_dataset('openbookqa', 'main')
    except Exception:
        try:
            ds = load_dataset('openbookqa')
        except Exception:
            ds = load_dataset('allenai/openbookqa', 'additional')
            
    test_split = ds['test'] if 'test' in ds else (ds['validation'] if 'validation' in ds else ds['train'])
    
    samples_data = []
    for idx in range(min(num_total, len(test_split))):
        s = test_split[idx]
        q = s.get('question_stem', s.get('question', ''))
        choices = s.get('choices', {})
        choices_text = choices.get('text', []) if isinstance(choices, dict) else (choices if isinstance(choices, list) else [])
        
        formatted_choices = ", ".join(choices_text) if choices_text else ""
        if formatted_choices:
            user_content = f"Choices: {formatted_choices}\nQuestion: {q}"
            fallback_text = f"Choices: {formatted_choices}\nQuestion: {q}\nAnswer:"
        else:
            user_content = q
            fallback_text = f"Question: {q}\nAnswer:"
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Based on the given choices, answer the question in a single sentence. Do not repeat the prompt."},
            {"role": "user", "content": user_content}
        ]
        try:
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_str = fallback_text
            
        samples_data.append({
            'idx': idx,
            'question': q,
            'prompt_str': prompt_str
        })
    return samples_data


def benchmark_single_model(model_name, args, deberta_model, deberta_tokenizer, deberta_embeddings, device):
    """
    Executes computational cost benchmark for a single LLM across all 5 Table 1 methods:
      1. Ours (Layer-Wise Ensembling)
      2. Kuhn SE (Semantic Entropy)
      3. SDLG (Semantically Diverse Substitutions)
      4. ICE (Input Clarification Ensembling)
      5. TokUR (Low-Rank Stochastic Weight Perturbation)
    """
    model_clean = model_name.split('/')[-1].lower().replace('-instruct', '').replace('-chat', '').replace('_', '-')
    output_prefix = getattr(args, 'output_dir_prefix', 'benchmark_cost_efficient')
    canonical_result_dir = os.path.join(experiments_dir, 'result')
    output_dir = os.path.join(canonical_result_dir, f'{output_prefix}_{args.dataset}_{model_clean}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" BENCHMARKING MODEL: {model_name}")
    print(f" Output Directory  : {output_dir}")
    print("=" * 80)

    # Ensure clean TokUR SVD cache state before initializing model
    clear_tokur_cache()

    # 1. Load LLM Tokenizer & Model
    is_phi = "phi-3" in model_name.lower() or "phi3" in model_name.lower()
    trust_remote = False if is_phi else True

    print(f"\n[1/2] Loading LLM Tokenizer & Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

    target_device_map = {"": 0} if (torch.cuda.is_available() and torch.cuda.device_count() == 1) else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        token=args.hf_token,
        trust_remote_code=trust_remote,
        device_map=target_device_map if torch.cuda.is_available() else None
    )
    model.eval()
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    total_layers = get_model_layer_count(model)

    # 2. Prepare Dataset Samples formatted with this model's tokenizer chat template
    total_needed = args.num_warmup + args.num_samples
    samples = load_openbookqa_prompts(tokenizer, total_needed)
    print(f"[2/2] Prepared {len(samples)} OpenBookQA prompts for {model_name}.\n")

    results = {
        'Ours': [],
        'Kuhn_SE': [],
        'SDLG': [],
        'ICE': [],
        'TokUR': []
    }

    # =========================================================================
    # METHOD 1: Ours (Layer-Wise Ensembling)
    # Pipeline: Greedy pass + Fused Forward Pass + DeBERTa attribution + Causal Prefix Slicing & Batched Layer Projection (up to S=10 tokens)
    # =========================================================================
    print("-" * 78)
    print(f">>> BENCHMARKING METHOD: Ours (Layer-Wise Ensembling) [{model_clean}]")
    print("-" * 78)

    for i, s in enumerate(tqdm(samples, desc=f"Ours [{model_clean}]")):
        is_warmup = (i < args.num_warmup)
        inputs = tokenizer(s['prompt_str'], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        def exec_ours():
            # Step A: Greedy generation
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            gen_ids = gen_out[0, prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            num_tokens = len(gen_ids)

            if num_tokens == 0 or len(gen_text.split()) == 0:
                return num_tokens, 0

            # Step B: Single Fused Forward Pass (Logits & Hidden States across all layers)
            with torch.no_grad():
                fwd_out = model(gen_out, return_dict=True, output_hidden_states=True)
                logits_out = fwd_out.logits[0, prompt_len-1:-1, :]
                gen_logits = torch.softmax(logits_out.float(), dim=-1) + 1e-9

            # Step C: Gradient backward pass on DeBERTa for salient token attribution
            args.alphas = (1/3, 1/3, 1/3)
            single_word = len(gen_text.split()) == 1
            if single_word:
                token_info = {}
                if args.token_prob_threshold is None:
                    other_voc_indices = torch.tensor(range(len(gen_logits[0])))
                else:
                    other_voc_indices = torch.where(gen_logits[0] > args.token_prob_threshold)[0]
                for new_token_idx in other_voc_indices:
                    if new_token_idx != gen_ids[0] and new_token_idx not in args.invalid_ids:
                        importance_score = gen_logits[0][new_token_idx]
                        token_info[(0, 0, new_token_idx.item())] = (0, 0, importance_score.item())
                if len(token_info) > 0:
                    sorted_indices = torch.argsort(rank_tensor([v[2] for v in token_info.values()], descending=True), descending=False)
                else:
                    sorted_indices = []
            else:
                additional_text = []
                sorted_indices, token_info = compute_token_score_ranking(
                    deberta_model, deberta_tokenizer, device,
                    s['question'], gen_ids.cpu(), gen_text,
                    additional_text, gen_logits.cpu(),
                    deberta_embeddings, tokenizer, args
                )

            # Step D: Causal Prefix Slicing & Batched Layer Projection across up to 10 important tokens
            token_info_list = list(token_info.keys()) if isinstance(token_info, dict) else []
            lm_head = model.get_output_embeddings()
            k_eval = min(16, total_layers - 1)
            cand_layers = list(range(total_layers - k_eval, total_layers))

            with torch.no_grad():
                eval_positions = []
                if (isinstance(sorted_indices, (torch.Tensor, list))) and len(token_info_list) > 0:
                    for s_idx in sorted_indices:
                        s_idx_val = s_idx.item() if hasattr(s_idx, 'item') else s_idx
                        if s_idx_val >= len(token_info_list):
                            continue
                        initial_gen_word_idx, initial_gen_token_idx, new_token_idx = token_info_list[s_idx_val]
                        token_to_replace_id = gen_ids[initial_gen_token_idx] if initial_gen_token_idx > 0 else gen_ids[0]

                        if new_token_idx in args.invalid_ids or new_token_idx == token_to_replace_id.item():
                            continue

                        # Causal position: In lower-triangular causal attention, position (prompt_len + token_idx - 1)
                        # contains the exact hidden state vector predicting token initial_gen_token_idx.
                        pos = prompt_len + initial_gen_token_idx - 1 if initial_gen_token_idx > 0 else prompt_len - 1
                        eval_positions.append(pos)
                        if len(eval_positions) >= args.num_total_generations:
                            break

                if not eval_positions:
                    eval_positions = [gen_out.shape[1] - 1]

                # Single batched LM-head kernel launch across all selected tokens and all layers
                target_dev = getattr(lm_head, 'weight', None).device if (hasattr(lm_head, 'weight') and hasattr(lm_head.weight, 'device')) else fwd_out.logits.device
                hs_stack = torch.stack([fwd_out.hidden_states[l][0, eval_positions, :].to(target_dev) for l in range(total_layers)], dim=0)
                # hs_stack shape: (total_layers, len(eval_positions), hidden_dim)
                all_logits = lm_head(hs_stack)
                # all_logits shape: (total_layers, len(eval_positions), vocab_size)

                # Vectorized pairwise mixture and divergence metric calculation per evaluated token
                for j, pos in enumerate(eval_positions):
                    final_logits = fwd_out.logits[0, pos, :].unsqueeze(0).float()
                    cand_logits = {l: all_logits[l, j].unsqueeze(0).to(final_logits.device) for l in cand_layers}

                    _ = calculate_all_disagreement_metrics(
                        final_logits, cand_logits,
                        relative_top=args.relative_top,
                        min_tokens_to_keep=args.min_tokens_to_keep,
                        skip_top_neighbors=args.skip_top_neighbors
                    )

            return num_tokens, 0

        (ans_tok, aux_tok), lat_s, alloc_gb, res_gb, dyn_mb = measure_gpu_execution(exec_ours)
        if not is_warmup:
            ms_per_ans_tok = (lat_s * 1000.0) / max(1, ans_tok)
            results['Ours'].append({
                'latency_s': lat_s,
                'ans_tokens': ans_tok,
                'aux_tokens': aux_tok,
                'ms_per_ans_token': ms_per_ans_tok,
                'peak_alloc_gb': alloc_gb,
                'peak_res_gb': res_gb,
                'dynamic_overhead_mb': dyn_mb
            })
            tqdm.write(f"  [Ours #{len(results['Ours'])}] Latency: {lat_s:.4f}s | Ans: {ans_tok} tok | {ms_per_ans_tok:.2f} ms/ans-tok | Peak: {alloc_gb:.2f}GB | Delta: +{dyn_mb:.1f}MB")

    # =========================================================================
    # METHOD 2: Kuhn SE (Semantic Entropy)
    # Pipeline: Greedy pass + S=10 sampling rollouts + S*(S-1)=90 DeBERTa NLI cross-encoder passes
    # =========================================================================
    print("\n" + "-" * 78)
    print(f">>> BENCHMARKING METHOD: Kuhn SE (Multi-Sequence Sampling + NLI Clustering) [{model_clean}]")
    print("-" * 78)
    torch.cuda.empty_cache()
    gc.collect()

    for i, s in enumerate(tqdm(samples, desc=f"Kuhn SE [{model_clean}]")):
        is_warmup = (i < args.num_warmup)
        inputs = tokenizer(s['prompt_str'], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        def exec_kuhn():
            with torch.no_grad():
                # 1. Greedy primary sequence
                greedy_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # 2. S = 10 stochastic sampling rollouts
                sampled_outs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=args.num_kuhn_samples,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                ans_tokens = (greedy_out.shape[1] - prompt_len)
                aux_tokens = (sampled_outs.shape[0] * (sampled_outs.shape[1] - prompt_len))
                
                # Decode sequences
                sampled_texts = tokenizer.batch_decode(sampled_outs[:, prompt_len:], skip_special_tokens=True)
                
                # 3. S * (S - 1) = 90 Pairwise DeBERTa NLI cross-encoder passes
                pairs = []
                for idx_a, ta in enumerate(sampled_texts):
                    for idx_b, tb in enumerate(sampled_texts):
                        if idx_a != idx_b:
                            text_a = f"{s['question']} {ta}"
                            text_b = f"{s['question']} {tb}"
                            pairs.append(f"{text_a} [SEP] {text_b}")
                
                if pairs:
                    for b_idx in range(0, len(pairs), 45):
                        chunk_pairs = pairs[b_idx:b_idx + 45]
                        nli_enc = deberta_tokenizer(chunk_pairs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                        _ = deberta_model(**nli_enc).logits
            return ans_tokens, aux_tokens

        (ans_tok, aux_tok), lat_s, alloc_gb, res_gb, dyn_mb = measure_gpu_execution(exec_kuhn)
        if not is_warmup:
            ms_per_ans_tok = (lat_s * 1000.0) / max(1, ans_tok)
            results['Kuhn_SE'].append({
                'latency_s': lat_s,
                'ans_tokens': ans_tok,
                'aux_tokens': aux_tok,
                'ms_per_ans_token': ms_per_ans_tok,
                'peak_alloc_gb': alloc_gb,
                'peak_res_gb': res_gb,
                'dynamic_overhead_mb': dyn_mb
            })
            tqdm.write(f"  [Kuhn #{len(results['Kuhn_SE'])}] Latency: {lat_s:.4f}s | Ans: {ans_tok} tok (+{aux_tok} aux) | {ms_per_ans_tok:.2f} ms/ans-tok | Peak: {alloc_gb:.2f}GB | Delta: +{dyn_mb:.1f}MB")

    # =========================================================================
    # METHOD 3: SDLG (Diverse Beam Search + Semantic Entropy)
    # Pipeline: Greedy pass + DeBERTa backward attribution + S=10 diverse substitutions + S*(S-1)=90 NLI passes
    # =========================================================================
    print("\n" + "-" * 78)
    print(f">>> BENCHMARKING METHOD: SDLG (Attribution + Diverse Substitutions + NLI Clustering) [{model_clean}]")
    print("-" * 78)
    torch.cuda.empty_cache()
    gc.collect()

    for i, s in enumerate(tqdm(samples, desc=f"SDLG [{model_clean}]")):
        is_warmup = (i < args.num_warmup)
        inputs = tokenizer(s['prompt_str'], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        def exec_sdlg():
            # 1. Greedy generation
            with torch.no_grad():
                greedy_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            gen_ids = greedy_out[0, prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            ans_tokens = len(gen_ids)

            with torch.no_grad():
                logits_out = model(greedy_out).logits[0, prompt_len-1:-1, :]
                gen_logits = torch.softmax(logits_out.float(), dim=-1) + 1e-9

            # 2. DeBERTa gradient backward attribution
            args.alphas = (1/3, 1/3, 1/3)
            sorted_indices, token_info = compute_token_score_ranking(
                deberta_model, deberta_tokenizer, device,
                s['question'], gen_ids.cpu(), gen_text,
                [], gen_logits.cpu(),
                deberta_embeddings, tokenizer, args
            )

            # 3. Generate S = 10 candidate trajectories via SDLG token substitution
            div_texts = [gen_text]
            aux_tokens = 0
            token_info_list = list(token_info.keys()) if isinstance(token_info, dict) else []

            with torch.no_grad():
                num_alts = min(args.num_sdlg_sequences - 1, len(token_info_list))
                for a_idx in range(num_alts):
                    _, _, new_tok_id = token_info_list[a_idx]
                    sub_prefix = torch.cat([inputs.input_ids, torch.tensor([[new_tok_id]], device=device)], dim=1)
                    alt_out = model.generate(
                        sub_prefix,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    alt_ids = alt_out[0, prompt_len:]
                    aux_tokens += len(alt_ids)
                    div_texts.append(tokenizer.decode(alt_ids, skip_special_tokens=True))

                # 4. Pairwise NLI cross-encoder passes (S * (S - 1) = 90 pairs)
                pairs = []
                for idx_a, ta in enumerate(div_texts):
                    for idx_b, tb in enumerate(div_texts):
                        if idx_a != idx_b:
                            pairs.append(f"{s['question']} {ta} [SEP] {s['question']} {tb}")
                if pairs:
                    for b_idx in range(0, len(pairs), 45):
                        chunk_pairs = pairs[b_idx:b_idx + 45]
                        nli_enc = deberta_tokenizer(chunk_pairs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                        _ = deberta_model(**nli_enc).logits
            return ans_tokens, aux_tokens

        (ans_tok, aux_tok), lat_s, alloc_gb, res_gb, dyn_mb = measure_gpu_execution(exec_sdlg)
        if not is_warmup:
            ms_per_ans_tok = (lat_s * 1000.0) / max(1, ans_tok)
            results['SDLG'].append({
                'latency_s': lat_s,
                'ans_tokens': ans_tok,
                'aux_tokens': aux_tok,
                'ms_per_ans_token': ms_per_ans_tok,
                'peak_alloc_gb': alloc_gb,
                'peak_res_gb': res_gb,
                'dynamic_overhead_mb': dyn_mb
            })
            tqdm.write(f"  [SDLG #{len(results['SDLG'])}] Latency: {lat_s:.4f}s | Ans: {ans_tok} tok (+{aux_tok} aux) | {ms_per_ans_tok:.2f} ms/ans-tok | Peak: {alloc_gb:.2f}GB | Delta: +{dyn_mb:.1f}MB")

    # =========================================================================
    # METHOD 4: ICE (Input Clarification Ensembling)
    # Pipeline: Greedy pass + M=5 clarification prompt generations + M=5 responses + 20 NLI passes
    # =========================================================================
    print("\n" + "-" * 78)
    print(f">>> BENCHMARKING METHOD: ICE (Clarification Generation + Response Ensembling) [{model_clean}]")
    print("-" * 78)
    torch.cuda.empty_cache()
    gc.collect()

    for i, s in enumerate(tqdm(samples, desc=f"ICE [{model_clean}]")):
        is_warmup = (i < args.num_warmup)
        inputs = tokenizer(s['prompt_str'], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        def exec_ice():
            with torch.no_grad():
                # 1. Greedy primary response
                greedy_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                ans_tokens = (greedy_out.shape[1] - prompt_len)
                aux_tokens = 0

                # 2. Stage 1: Generate M = 5 clarification queries
                clarif_prompt = f"{s['prompt_str']} Generate clarification question:"
                c_inputs = tokenizer(clarif_prompt, return_tensors="pt").to(device)
                c_len = c_inputs.input_ids.shape[1]
                clarif_outs = model.generate(
                    **c_inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=args.num_ice_clarifications,
                    pad_token_id=tokenizer.pad_token_id
                )
                aux_tokens += (clarif_outs.shape[0] * (clarif_outs.shape[1] - c_len))
                clarif_texts = tokenizer.batch_decode(clarif_outs[:, c_len:], skip_special_tokens=True)

                # 3. Stage 2: Generate response for each clarification
                resp_texts = []
                for c_txt in clarif_texts:
                    resp_prompt = f"{s['prompt_str']} Clarification: {c_txt}\nAnswer:"
                    r_inputs = tokenizer(resp_prompt, return_tensors="pt").to(device)
                    r_len = r_inputs.input_ids.shape[1]
                    resp_out = model.generate(
                        **r_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    aux_tokens += (resp_out.shape[1] - r_len)
                    resp_texts.append(tokenizer.decode(resp_out[0, r_len:], skip_special_tokens=True))

                # 4. Stage 3: Pairwise DeBERTa NLI cross-encoder passes (M * (M - 1) = 20 pairs)
                pairs = []
                for idx_a, ta in enumerate(resp_texts):
                    for idx_b, tb in enumerate(resp_texts):
                        if idx_a != idx_b:
                            pairs.append(f"{s['question']} {ta} [SEP] {s['question']} {tb}")
                if pairs:
                    nli_enc = deberta_tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    _ = deberta_model(**nli_enc).logits
            return ans_tokens, aux_tokens

        (ans_tok, aux_tok), lat_s, alloc_gb, res_gb, dyn_mb = measure_gpu_execution(exec_ice)
        if not is_warmup:
            ms_per_ans_tok = (lat_s * 1000.0) / max(1, ans_tok)
            results['ICE'].append({
                'latency_s': lat_s,
                'ans_tokens': ans_tok,
                'aux_tokens': aux_tok,
                'ms_per_ans_token': ms_per_ans_tok,
                'peak_alloc_gb': alloc_gb,
                'peak_res_gb': res_gb,
                'dynamic_overhead_mb': dyn_mb
            })
            tqdm.write(f"  [ICE #{len(results['ICE'])}] Latency: {lat_s:.4f}s | Ans: {ans_tok} tok (+{aux_tok} aux) | {ms_per_ans_tok:.2f} ms/ans-tok | Peak: {alloc_gb:.2f}GB | Delta: +{dyn_mb:.1f}MB")

    # =========================================================================
    # METHOD 5: TokUR (Token-Level Low-Rank Weight Perturbation)
    # Pipeline: Greedy pass + P=5 forward passes under LowRankBayesianLinear
    # Executed LAST so modifying the model's nn.Linear modules does not affect prior methods
    # =========================================================================
    print("\n" + "-" * 78)
    print(f">>> BENCHMARKING METHOD: TokUR (Low-Rank Weight Perturbation) [{model_clean}]")
    print("-" * 78)
    torch.cuda.empty_cache()
    gc.collect()

    print("[INFO] Wrapping model linear layers with LowRankBayesianLinear (r=4, sigma=0.05)...")
    t_svd_start = time.perf_counter()
    bayesian_layers = wrap_model_with_tokur(
        model=model,
        rank=args.tokur_rank,
        sigma=args.tokur_sigma,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]
    )
    svd_setup_time_s = time.perf_counter() - t_svd_start
    print(f"[INFO] Wrapped {len(bayesian_layers)} linear modules in {svd_setup_time_s:.2f}s CPU SVD initialization.")

    for i, s in enumerate(tqdm(samples, desc=f"TokUR [{model_clean}]")):
        is_warmup = (i < args.num_warmup)
        inputs = tokenizer(s['prompt_str'], return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        def exec_tokur():
            with torch.no_grad():
                # 1. Unperturbed greedy generation
                set_model_sampling(bayesian_layers, sample=False)
                greedy_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                ans_tokens = greedy_out.shape[1] - prompt_len

                # 2. P = 5 stochastic perturbed forward passes
                set_model_sampling(bayesian_layers, sample=True)
                for _ in range(args.num_tokur_perturbs):
                    _ = model(greedy_out).logits
            return ans_tokens, 0

        (ans_tok, aux_tok), lat_s, alloc_gb, res_gb, dyn_mb = measure_gpu_execution(exec_tokur)
        if not is_warmup:
            ms_per_ans_tok = (lat_s * 1000.0) / max(1, ans_tok)
            results['TokUR'].append({
                'latency_s': lat_s,
                'ans_tokens': ans_tok,
                'aux_tokens': aux_tok,
                'ms_per_ans_token': ms_per_ans_tok,
                'peak_alloc_gb': alloc_gb,
                'peak_res_gb': res_gb,
                'dynamic_overhead_mb': dyn_mb
            })
            tqdm.write(f"  [TokUR #{len(results['TokUR'])}] Latency: {lat_s:.4f}s | Ans: {ans_tok} tok | {ms_per_ans_tok:.2f} ms/ans-tok | Peak: {alloc_gb:.2f}GB | Delta: +{dyn_mb:.1f}MB")

    # =========================================================================
    # MODEL REPORT & EXPORT
    # =========================================================================
    SEQ_CONFIGS = {
        'Ours': f'{args.num_total_generations} tokens (causal)',
        'TokUR': f'{args.num_tokur_perturbs} perturbs',
        'Kuhn_SE': f'{args.num_kuhn_samples} samples',
        'SDLG': f'{args.num_sdlg_sequences} seqs',
        'ICE': f'{args.num_ice_clarifications * 2} queries'
    }

    mean_ans_len = np.mean([r['ans_tokens'] for r in results['Ours']]) if results['Ours'] else 8.5

    print("\n" + "=" * 132)
    print(f" BENCHMARK SUMMARY: {model_name} ON {args.dataset.upper()} (N = {args.num_samples})")
    print(f"  * Generated Answer Length : ~{mean_ans_len:.1f} tokens (invariant primary answer to query)")
    print(f"  * Delta VRAM (UQ Overhead): Dynamic runtime memory allocated above static base model weights")
    print(f"  * SVD Setup Overhead      : {svd_setup_time_s:.2f}s one-time CPU pre-computation across {len(bayesian_layers)} layers (TokUR only)")
    print("=" * 132)
    header = f"{'Method':<10} | {'Sequences':<16} | {'Query Lat. (s)':<18} | {'ms / ans tok':<15} | {'Throughput':<12} | {'vs. Ours':<10} | {'Peak VRAM':<11} | {'Delta VRAM':<15} | {'SVD Setup':<10}"
    print(header)
    print("-" * 132)

    summary_rows = []
    ours_mean_lat = np.mean([r['latency_s'] for r in results['Ours']]) if results['Ours'] else 1.0

    for name in ['Ours', 'TokUR', 'Kuhn_SE', 'SDLG', 'ICE']:
        data = results[name]
        if not data:
            continue
        lats = [r['latency_s'] for r in data]
        ans_toks = [r['ans_tokens'] for r in data]
        aux_toks = [r['aux_tokens'] for r in data]
        ms_toks = [r['ms_per_ans_token'] for r in data]
        allocs = [r['peak_alloc_gb'] for r in data]
        dyns = [r['dynamic_overhead_mb'] for r in data]

        mean_lat, std_lat = np.mean(lats), np.std(lats)
        mean_ans, std_ans = np.mean(ans_toks), np.std(ans_toks)
        mean_aux = np.mean(aux_toks)
        mean_ms_tok, std_ms_tok = np.mean(ms_toks), np.std(ms_toks)
        qps = 1.0 / max(1e-5, mean_lat)
        max_alloc = np.max(allocs)
        mean_dyn = np.mean(dyns)
        seq_label = SEQ_CONFIGS.get(name, '-')
        slowdown = mean_lat / max(1e-5, ours_mean_lat)
        svd_str = f"{svd_setup_time_s:.1f}s" if name == 'TokUR' else "0.0s"

        print(f"{name:<10} | {seq_label:<16} | {mean_lat:5.3f} ± {std_lat:4.3f} s    | {mean_ms_tok:5.2f} ± {std_ms_tok:4.2f} ms | {qps:5.2f} q/s   | {slowdown:4.2f}x      | {max_alloc:5.2f} GB    | +{mean_dyn:6.1f} MB       | {svd_str:<10}")
        
        row_dict = {
            'Model': model_clean,
            'Method': name,
            'Sequences': seq_label,
            'Mean_Latency_s': round(float(mean_lat), 4),
            'Std_Latency_s': round(float(std_lat), 4),
            'Mean_Ans_Tokens': round(float(mean_ans), 2),
            'Mean_Aux_Tokens': round(float(mean_aux), 2),
            'Mean_ms_per_ans_token': round(float(mean_ms_tok), 2),
            'Std_ms_per_ans_token': round(float(std_ms_tok), 2),
            'Throughput_qps': round(float(qps), 2),
            'Slowdown_vs_Ours': round(float(slowdown), 2),
            'Peak_Allocated_VRAM_GB': round(float(max_alloc), 2),
            'Delta_VRAM_MB': round(float(mean_dyn), 2),
            'SVD_Setup_Time_s': round(float(svd_setup_time_s), 2) if name == 'TokUR' else 0.0
        }
        summary_rows.append(row_dict)

    print("=" * 132)

    # Save to model-specific CSV in canonical result directory
    csv_path = os.path.join(output_dir, 'benchmark_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[SAVED] Benchmark CSV exported to: {csv_path}")

    # Save detailed JSON
    json_path = os.path.join(output_dir, 'benchmark_details.json')
    meta_info = {
        'model_name': model_name,
        'model_clean': model_clean,
        'svd_init_time_s': svd_setup_time_s,
        'total_layers': total_layers,
        'args': {k: str(v) for k, v in args.__dict__.items()}
    }
    with open(json_path, 'w') as f:
        json.dump({'meta': meta_info, 'summary': summary_rows, 'raw_results': results}, f, indent=2)
    print(f"[SAVED] Detailed JSON exported to: {json_path}")

    # Explicit memory cleanup before next model
    del model, tokenizer
    if 'bayesian_layers' in locals():
        del bayesian_layers
    clear_tokur_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_rows


def run_benchmark():
    args = BenchmarkCostArgs()
    args.alphas = (1/3, 1/3, 1/3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.run_model_indices is not None:
        models_to_run = [args.models_to_benchmark[idx] for idx in args.run_model_indices]
    else:
        models_to_run = list(args.models_to_benchmark)

    print("=" * 100)
    print(" COMPUTATIONAL COST BENCHMARK SUITE (TABLE 1 ALL 9 MODELS)")
    print(f" Dataset : {args.dataset.upper()} (Consistent Single-Sentence Generation)")
    print(f" Models  : {len(models_to_run)} model(s) queued for sequential benchmarking")
    print(f" Samples : {args.num_samples} (plus {args.num_warmup} initial warmup queries)")
    print(f" Device  : {device}")
    print("=" * 100)

    # Load DeBERTa ONCE upfront (shared across all models for NLI & attribution)
    print(f"\n[INIT] Loading shared DeBERTa model: {args.deberta_model}...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(args.deberta_model, token=args.hf_token)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(args.deberta_model, token=args.hf_token).to(device)
    deberta_model.eval()

    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))], device=device)
    ).squeeze().detach()
    print("[INIT] DeBERTa loaded successfully and cached.")

    all_models_summaries = []
    canonical_result_dir = os.path.join(experiments_dir, 'result')

    for m_idx, model_name in enumerate(models_to_run):
        print("\n" + "#" * 100)
        print(f" MODEL [{m_idx + 1}/{len(models_to_run)}]: {model_name}")
        print("#" * 100)

        model_clean = model_name.split('/')[-1].lower().replace('-instruct', '').replace('-chat', '').replace('_', '-')
        output_prefix = getattr(args, 'output_dir_prefix', 'benchmark_cost_efficient')
        model_out_dir = os.path.join(canonical_result_dir, f'{output_prefix}_{args.dataset}_{model_clean}')
        summary_csv = os.path.join(model_out_dir, 'benchmark_summary.csv')

        if getattr(args, 'skip_completed', True) and os.path.exists(summary_csv) and not getattr(args, 'force_rerun', False):
            try:
                with open(summary_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    if 'Mean_ms_per_ans_token' in fieldnames and 'Sequences' in fieldnames:
                        print(f"[SKIP/RESUME] Valid up-to-date benchmark summary found at: {summary_csv}")
                        print(f"              Skipping {model_name} (set force_rerun=True in BenchmarkCostArgs to re-run).")
                        for row in reader:
                            all_models_summaries.append(row)
                        continue
                    else:
                        print(f"[INFO] Existing summary for {model_name} is from the legacy schema (missing normalized answer token metrics).")
                        print(f"       Re-running {model_name} to generate the updated publication-grade schema...")
            except Exception as e:
                print(f"[WARN] Failed to read existing summary CSV: {e}. Re-running...")

        try:
            summary_rows = benchmark_single_model(
                model_name=model_name,
                args=args,
                deberta_model=deberta_model,
                deberta_tokenizer=deberta_tokenizer,
                deberta_embeddings=deberta_embeddings,
                device=device
            )
            all_models_summaries.extend(summary_rows)
        except Exception as e:
            print(f"\n[ERROR] Failed benchmarking model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Continuing to next model in queue...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Consolidated Master Export
    if all_models_summaries:
        output_prefix = getattr(args, 'output_dir_prefix', 'benchmark_cost_efficient')
        master_csv_path = os.path.join(canonical_result_dir, f'{output_prefix}_{args.dataset}_all_9_models_summary.csv')
        os.makedirs(os.path.dirname(master_csv_path), exist_ok=True)
        with open(master_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_models_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_models_summaries)
        print("\n" + "=" * 136)
        print(f" MASTER CONSOLIDATED BENCHMARK SUMMARY (ALL EVALUATED MODELS)")
        print(f" Master CSV: {master_csv_path}")
        print("=" * 136)
        print(f"{'Model':<20} | {'Method':<10} | {'Sequences':<14} | {'Query Latency (s)':<18} | {'ms / ans tok':<14} | {'Throughput':<10} | {'vs. Ours':<9} | {'Peak VRAM':<10} | {'Delta VRAM':<13} | {'SVD Setup':<10}")
        print("-" * 136)
        for r in all_models_summaries:
            print(f"{r.get('Model',''):<20} | {r.get('Method',''):<10} | {r.get('Sequences','-'):<14} | {float(r.get('Mean_Latency_s',0)):5.3f} ± {float(r.get('Std_Latency_s',0)):4.3f} s   | {float(r.get('Mean_ms_per_ans_token',0)):5.2f} ms/tok   | {float(r.get('Throughput_qps',0)):4.2f} q/s   | {float(r.get('Slowdown_vs_Ours',1)):4.2f}x    | {float(r.get('Peak_Allocated_VRAM_GB',0)):5.2f} GB   | +{float(r.get('Delta_VRAM_MB',0)):5.1f} MB     | {float(r.get('SVD_Setup_Time_s',0)):.1f}s")
        print("=" * 136)


if __name__ == '__main__':
    run_benchmark()
