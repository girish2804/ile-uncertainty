import os
import sys
import importlib.util

# 1. TF & CUDA safety flags BEFORE PyTorch imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 2. DynamicCache Compatibility Shim for Phi-3 / DeepSeek MoE ('seen_tokens', 'get_max_length' errors)
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

import argparse
import pickle
import gzip
import yaml
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
import traceback

# Add src/ to Python path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layer_ensemble import generate_layer_ensemble_sequences, generate_semantically_diverse_output_sequences
from utils import seed_everything, generate_text, compute_likelihood, compute_correctness, compute_semantic_paris_new
from args import Args

def run_experiment():
    args = Args()
    
    # Configure defaults on the args object if not already defined in args.py
    if not hasattr(args, 'start_idx'):
        args.start_idx = 0
    if not hasattr(args, 'num_samples'):
        args.num_samples = -1  # Number of samples to process; set to -1 for all
        
    # Standardize default values if they are the placeholder defaults in args.py
    if args.llm_model == 'llama-7b':
        args.llm_model = "Qwen/Qwen2.5-1.5B-Instruct"
        
    if not hasattr(args, 'run_id') or args.run_id in ['truthfulqa_llama-7b', 'truthfulqa_qwen2.5-1.5b']:
        args.run_id = getattr(args, 'dynamic_run_id', args.run_id)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'result', args.run_id)
    os.makedirs(base_path, exist_ok=True)
    
    # Save current run configuration
    args.args_to_yaml(base_path)
    print("=" * 60)
    print(f"RUNNING TRUTHFUL_QA EXPERIMENT: {args.run_id}")
    print(f"LLM: {args.llm_model}")
    print(f"DeBERTa: {args.deberta_model}")
    print(f"Results Directory: {base_path}")
    print("=" * 60)
    
    seed_everything(seed=args.seed_value)
    
    # Configure devices
    device_llm = "cuda" if torch.cuda.is_available() else "cpu"
    device_deberta = "cuda" if torch.cuda.is_available() else "cpu"
    
    auth_token = getattr(args, 'hf_token', None) or os.environ.get("HF_TOKEN", None)
    
    skip_gen = getattr(args, 'skip_generation', False)
    
    if not skip_gen:
        # 1. Load LLM FIRST so it has the entire GPU memory available
        print(f"Loading LLM Model and Tokenizer: {args.llm_model}...")
        is_phi = "phi-3" in str(args.llm_model).lower() or "phi3" in str(args.llm_model).lower()
        trust_remote = False if is_phi else True
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, token=auth_token, trust_remote_code=trust_remote)
        
        # Setup quantization config if requested
        quantization_config = None
        quant_mode = getattr(args, 'quantization', None)
        if quant_mode in ['4bit', '8bit'] and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            if quant_mode == '4bit':
                print("Configuring 4-bit (NF4) quantization (fp32 CPU offload enabled)...")
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
        
        attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
        if torch.cuda.is_available():
            # For single-GPU setups, {"": 0} instructs Transformers directly to keep all layers on GPU 0
            # and prevents accelerate's heuristic infer_auto_device_map from conservatively dispatching modules to CPU.
            target_device_map = {"": 0} if torch.cuda.device_count() == 1 else "auto"
            print(f"Loading LLM with device_map={target_device_map} (GPUs available: {torch.cuda.device_count()})...")

            llm_model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                token=auth_token,
                trust_remote_code=trust_remote,
                attn_implementation=attn_impl,
                device_map=target_device_map
            )
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                torch_dtype=torch.float32,
                token=auth_token,
                trust_remote_code=trust_remote,
                attn_implementation=attn_impl
            ).to(device_llm)
            
        device_llm = next(llm_model.parameters()).device
        print(f"LLM loaded successfully on {device_llm}")
        
        # Flush VRAM allocation artifacts before loading DeBERTa
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Load DeBERTa
    print(f"Loading DeBERTa Model and Tokenizer: microsoft/{args.deberta_model}...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{args.deberta_model}", token=auth_token)
    device_deberta = "cuda:0" if torch.cuda.is_available() else "cpu"
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{args.deberta_model}",
        token=auth_token
    ).to(device_deberta)
    print(f"DeBERTa loaded successfully on {device_deberta}")

    if not skip_gen:
        # Extract DeBERTa word embeddings
        print("Extracting DeBERTa word embeddings matrix...")
        deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
            torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
        ).squeeze().detach()
        
        # Setup pad/eos tokens safely
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                llm_model.resize_token_embeddings(len(tokenizer))
                
        llm_model.config.pad_token_id = tokenizer.pad_token_id
        
        # Setup robust EOS token list
        eos_ids = []
        if tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                eos_ids.extend(tokenizer.eos_token_id)
            else:
                eos_ids.append(tokenizer.eos_token_id)
        if hasattr(llm_model, 'config') and llm_model.config.eos_token_id is not None:
            if isinstance(llm_model.config.eos_token_id, list):
                eos_ids.extend(llm_model.config.eos_token_id)
            else:
                eos_ids.append(llm_model.config.eos_token_id)
        for token_str in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|end|>", "\n\n"]:
            try:
                enc_ids = tokenizer.encode(token_str, add_special_tokens=False)
                for tid in enc_ids:
                    if tid not in eos_ids:
                        eos_ids.append(tid)
            except Exception:
                pass
        args.eos_token_ids = list(set(eos_ids))
        args.invalid_ids = getattr(args, 'invalid_ids', []) or []
        
        # Load TruthfulQA dataset
        print("Loading truthfulqa (generation split) dataset...")
        try:
            dataset = load_dataset('truthfulqa/truthful_qa', 'generation')
        except Exception:
            dataset = load_dataset('truthful_qa', 'generation')
        val_set = dataset['validation']
        total_dataset_len = len(val_set)
        print(f"Dataset loaded. Total validation samples: {total_dataset_len}")
        
        # Define validation slice
        start_idx = args.start_idx
        num_samples = args.num_samples
        if num_samples == -1 or start_idx + num_samples > total_dataset_len:
            num_samples = total_dataset_len - start_idx
            
        end_idx = start_idx + num_samples
        print(f"Processing slice: indices {start_idx} to {end_idx} ({num_samples} samples)")
        
        # Load Evaluation Metrics
        rouge = evaluate.load('rouge')
        exact_match_metric = evaluate.load("exact_match")
        bleurt = None
    else:
        # Configuration for evaluation-only mode
        start_idx = args.start_idx
        num_samples = args.num_samples
        if num_samples == -1:
            import glob
            existing_files = glob.glob(os.path.join(base_path, 'results_dict_*.*'))
            max_idx = -1
            for f in existing_files:
                basename = os.path.basename(f)
                try:
                    idx_str = basename.replace('results_dict_', '').split('.')[0]
                    val = int(idx_str)
                    if val > max_idx:
                        max_idx = val
                except ValueError:
                    continue
            if max_idx >= start_idx:
                num_samples = max_idx + 1 - start_idx
            else:
                num_samples = 0
        end_idx = start_idx + num_samples
        total_dataset_len = num_samples
        print(f"Skipping LLM text generation. Will calculate metrics for indices {start_idx} to {end_idx} ({num_samples} samples)")
        
    # Main Generation and Layer Ensembling Loop
    processed_count = 0
    failed_indices = []
    if not skip_gen:
        # Find the last completed index to resume quickly
        max_existing_idx = -1
        import glob
        existing_files = glob.glob(os.path.join(base_path, 'results_dict_*.*'))
        for f in existing_files:
            basename = os.path.basename(f)
            try:
                idx_str = basename.replace('results_dict_', '').split('.')[0]
                val = int(idx_str)
                if val > max_existing_idx:
                    max_existing_idx = val
            except ValueError:
                continue
                
        resume_idx = start_idx
        if max_existing_idx >= start_idx:
            # Check only the last file for corruption
            last_file_corrupt = False
            for ext in ['.pkl.gz', '.pkl']:
                file_path = os.path.join(base_path, f'results_dict_{max_existing_idx}{ext}')
                if os.path.exists(file_path):
                    try:
                        if ext == '.pkl.gz':
                            with gzip.open(file_path, 'rb') as f:
                                pickle.load(f)
                        else:
                            with open(file_path, 'rb') as f:
                                pickle.load(f)
                    except Exception:
                        last_file_corrupt = True
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
            if last_file_corrupt:
                resume_idx = max(start_idx, max_existing_idx)
                print(f"Detected corrupt last file at index {max_existing_idx}. Resuming from index {resume_idx}.")
            else:
                resume_idx = max(start_idx, max_existing_idx + 1)
                print(f"Resuming quickly from index {resume_idx} (last completed index: {max_existing_idx}).")
                
        processed_count = max(0, resume_idx - start_idx)
        
        for idx in tqdm(range(start_idx, end_idx), desc="Running experiment generations"):
            out_file_gz = os.path.join(base_path, f'results_dict_{idx}.pkl.gz')
            out_file_raw = os.path.join(base_path, f'results_dict_{idx}.pkl')
            if (os.path.exists(out_file_gz) or os.path.exists(out_file_raw)) and not getattr(args, 'force_overwrite', False):
                continue
            try:

                sample = val_set[idx]
                question = sample['question']
                best_answer = sample['best_answer']
                correct_answers = sample.get('correct_answers', [best_answer])
                incorrect_answers = sample.get('incorrect_answers', [])
                
                # Format using modern chat templates if supported
                messages = [
                    {"role": "system", "content": "Provide a direct, concise, single-sentence answer to the user's question. Do not repeat system instructions, do not repeat the prompt, do not list multiple options, and do not explain your answer."},
                    {"role": "user", "content": question}
                ]
                try:
                    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt_text = f"Answer the following question directly and concisely in a single sentence:\nQuestion: {question}\nAnswer:"
                    
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device_llm)
                prompt_ids = inputs["input_ids"][0].to('cpu')
                
                # 1. Greedy Generation
                most_likely_generation = generate_text(
                    args=args,
                    model=llm_model,
                    tokenizer=tokenizer,
                    input_ids=inputs["input_ids"],
                    len_prompt=len(prompt_ids),
                    decoding_method="most_likely",
                    device=device_llm
                )
                
                # 2. Compute Correctness Metric
                correctness_dict = compute_correctness(
                    args=args,
                    reference_answers=correct_answers,
                    incorrect_answers=incorrect_answers,
                    most_likely_generation_text=most_likely_generation['generation_text'][0],
                    exact_match_metric=exact_match_metric,
                    rouge=rouge,
                    bleurt=bleurt
                )
                
                # 3. Compute likelihood of initial response
                most_likely_generation_likelihoods = compute_likelihood(
                    prompt=prompt_ids,
                    generation=most_likely_generation,
                    model=llm_model,
                    device=device_llm,
                    compute_cleaned=args.compute_cleaned,
                    store_logits=True
                )
                
                results_dict = {
                    'input_ids': inputs['input_ids'].to('cpu'),
                    'question': question,
                    'question_id': idx,
                    'best_answer': best_answer,
                    'correct_answers': correct_answers,
                    'incorrect_answers': incorrect_answers,
                    'correctness_dict': correctness_dict,
                    'sdlg': {
                        'generations': [most_likely_generation],
                        'likelihoods': [most_likely_generation_likelihoods],
                        'epistem_entropies': []
                    }
                }
                
                # 4. Intermediate Layer Ensembling & Salient Token Evaluation (Attribution via SDLG)
                results_dict = generate_semantically_diverse_output_sequences(
                    results_dict=results_dict,
                    deberta_model=deberta_model,
                    deberta_tokenizer=deberta_tokenizer,
                    device_deberta=device_deberta,
                    deberta_embeddings=deberta_embeddings,
                    model=llm_model,
                    tokenizer=tokenizer,
                    device_llm=device_llm,
                    input_ids=inputs['input_ids'],
                    prompt=prompt_ids,
                    question=question,
                    initial_generation=most_likely_generation,
                    initial_likelihood=most_likely_generation_likelihoods,
                    key='sdlg',
                    compute_epistemic=True,
                    args=args
                )
                
                # Strip logits if store_logits is disabled to save disk space
                if not args.store_logits:
                    for key in ['sdlg']:
                        if key in results_dict:
                            for lik in results_dict[key].get('likelihoods', []):
                                if 'generation_logits' in lik:
                                    lik['generation_logits'] = []
                                if 'cleaned_generation_logits' in lik:
                                    lik['cleaned_generation_logits'] = []
                                    
                # Save results
                if args.compress_pkl:
                    save_file = os.path.join(base_path, f'results_dict_{idx}.pkl.gz')
                    with gzip.open(save_file, 'wb') as outfile:
                        pickle.dump(results_dict, outfile)
                else:
                    save_file = os.path.join(base_path, f'results_dict_{idx}.pkl')
                    with open(save_file, 'wb') as outfile:
                        pickle.dump(results_dict, outfile)
                        
                # Print progress in real-time
                rougel_val = correctness_dict.get('rougeL', 0.0)
                em_val = correctness_dict.get('exact_match', 0.0)
                
                tqdm.write(f"\n[Sample #{idx}] Question: {question}")
                tqdm.write(f"  Greedy Answer : '{most_likely_generation['generation_text'][0]}'")
                tqdm.write(f"  Reference Ans : {correct_answers}")
                tqdm.write(f"  Correctness   : ROUGE-L = {rougel_val:.4f} | EM = {em_val:.1f} (BLEURT computed in batched pass at end)")
                tqdm.write(f"  SDLG Alts ({len(results_dict['sdlg']['generations']) - 1}):")
                for g_idx, gen in enumerate(results_dict['sdlg']['generations']):
                    if g_idx > 0:
                        tqdm.write(f"   [{g_idx}] '{gen['generation_text'][0]}'")
                tqdm.write("-" * 60)
    
                processed_count += 1
                
                # Free VRAM to prevent leaks and fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            except Exception as e:
                failed_indices.append((idx, str(e)))
                print(f"Error processing sample index {idx}: {e}")
                traceback.print_exc()
                if "CUDA error" in str(e) or "illegal memory access" in str(e):
                    print("Fatal CUDA error encountered. Exiting script to prevent terminal flooding.")
                    sys.exit(1)
                
    if failed_indices:
        print("\n" + "!" * 70)
        print(f"[ALERT] {len(failed_indices)} samples failed during generation:")
        for f_idx, f_err in failed_indices:
            print(f"  - Sample #{f_idx}: {f_err}")
        print("!" * 70 + "\n")

    # Verify if valid result files exist in base_path
    existing_results = glob.glob(os.path.join(base_path, 'results_dict_*.pkl*'))
    valid_results = [f for f in existing_results if not os.path.basename(f).startswith('results_dict_semantic_matrices_')]
    if len(valid_results) == 0:
        print(f"\n[FATAL ERROR] 0 samples were generated and no prior results found in {base_path}.")
        print("Aborting downstream semantic clustering and BLEURT evaluation.")
        sys.exit(1)

    print(f"\nCompleted generations. Successfully ready samples: {len(valid_results)}.")

    # 5. Free LLM GPU memory immediately after generation
    print("\n[CLEANUP] Freeing LLM GPU memory before semantic pairs clustering...")
    if not skip_gen:
        if 'llm_model' in locals():
            del llm_model
        if 'deberta_embeddings' in locals():
            del deberta_embeddings
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 6. Automatically compute NLI semantic equivalence and sequence-level importance scores
    if len(valid_results) > 0:
        print("\nComputing semantic pairs and sequence importance scores...")
        try:
            compute_semantic_paris_new(
                base_path=base_path,
                model_type=args.deberta_model,
                deberta_tokenizer=deberta_tokenizer,
                deberta_model=deberta_model,
                num_instances=num_samples,
                device=device_deberta,
                offset=start_idx
            )
            print("Semantic pairs computation completed successfully!")
        except Exception as e:
            print(f"Error computing semantic pairs: {e}")
            traceback.print_exc()

    # 7. Free DeBERTa GPU memory before batched BLEURT
    print("\n[CLEANUP] Freeing DeBERTa GPU memory before batched BLEURT...")
    if 'deberta_model' in locals():
        del deberta_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 8. Fast Vectorized Batched BLEURT Scoring strictly against single best_answer
    from batch_scoring import run_batched_bleurt_scoring
    bleurt_batch_sz = getattr(args, 'bleurt_batch_size', 64)
    run_batched_bleurt_scoring(base_path=base_path, batch_size=bleurt_batch_sz)

    # 9. Auto-Trigger Batch Clustering & 3xN Pickle Export to result/entropies_unified/
    if getattr(args, 'auto_batch_compute', True):
        from batch_compute import run_batch_compute
        target_dir = getattr(args, 'entropies_dir', 'entropies_unified')
        run_batch_compute(args=args, base_path=base_path, target_entropies_dir=target_dir)

    status_str = "SUCCESS (with warnings)" if failed_indices else "SUCCESS"
    print("=" * 70)
    print(f"EXPERIMENT RUN FINISHED [{status_str}]:")
    print(f"  - Dataset                 : TruthfulQA ({total_dataset_len} total instances)")
    print(f"  - Valid Processed Samples : {len(valid_results)}")
    print(f"  - Failed Samples in Loop  : {len(failed_indices)}")
    print(f"  - Raw Sample Directory    : {base_path}")
    print(f"  - Exported Entropies Dir  : {os.path.join(script_dir, 'result', getattr(args, 'entropies_dir', 'entropies_unified'))}")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
