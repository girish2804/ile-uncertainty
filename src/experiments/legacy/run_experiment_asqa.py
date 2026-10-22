import os
import sys
import gc
import glob
import gzip
import pickle
import traceback
import numpy as np
from tqdm import tqdm

# Ensure TensorFlow pre-allocation safety flags
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate

# Add src/ to Python path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from layer_ensemble import generate_layer_ensemble_sequences, generate_semantically_diverse_output_sequences
from utils import (
    seed_everything, 
    generate_text, 
    compute_likelihood, 
    compute_correctness, 
    compute_semantic_paris_new
)
from asqa_args import ASQAArgs


def load_asqa_samples(split='dev', num_samples=-1):
    """
    Loads and standardizes the ASQA dataset (din0s/asqa) from Hugging Face.
    Extracts ambiguous_question as the prompt and human-written long_answer
    annotations as the symmetric ground-truth references for BLEURT evaluation.
    """
    print(f"[INFO] Loading ASQA benchmark dataset (split='{split}')...")
    try:
        ds = load_dataset("din0s/asqa", split=split)
    except Exception as e:
        print(f"[ERROR] Failed to load din0s/asqa directly from Hugging Face: {e}")
        raise e

    samples = []
    total_len = len(ds)
    limit = total_len if num_samples <= 0 else min(num_samples, total_len)

    for i in range(limit):
        item = ds[i]
        sample_id = item.get('sample_id', str(i))
        question = item.get('ambiguous_question', '').strip()
        annotations = item.get('annotations', [])
        qa_pairs = item.get('qa_pairs', [])

        # Collect human-written long-form reference answers
        long_answers = []
        for anno in annotations:
            if isinstance(anno, dict):
                la = anno.get('long_answer', '').strip()
                if la and la not in long_answers:
                    long_answers.append(la)

        # Fallback if long_answers is somehow empty: aggregate qa_pairs short answers
        if not long_answers and qa_pairs:
            flat_shorts = []
            for qap in qa_pairs:
                for sa in qap.get('short_answers', []):
                    if sa.strip() and sa.strip() not in flat_shorts:
                        flat_shorts.append(sa.strip())
            if flat_shorts:
                long_answers.append("; ".join(flat_shorts))

        best_answer = long_answers[0] if long_answers else ""

        samples.append({
            'id': sample_id,
            'question': question,
            'best_answer': best_answer,
            'correct_answers': long_answers,
            'incorrect_answers': [],
            'qa_pairs': qa_pairs,
            'annotations': annotations
        })

    print(f"[INFO] Loaded {len(samples)} valid ASQA samples.")
    return samples


def run_experiment():
    args = ASQAArgs()
    
    # Configure base path
    model_name_clean = args.llm_model.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "").replace("_", "-")
    args.run_id = f"asqa-{model_name_clean}"
    base_path = os.path.join(script_dir, 'result', args.run_id)
    os.makedirs(base_path, exist_ok=True)
    
    # Save run arguments
    args.args_to_yaml(base_path)
    
    print("=" * 70)
    print(f"RUNNING ASQA BENCHMARK EXPERIMENT: {args.run_id}")
    print(f"Dataset         : ASQA (din0s/asqa, split: {args.split})")
    print(f"LLM Model       : {args.llm_model}")
    print(f"DeBERTa Model   : microsoft/{args.deberta_model}")
    print(f"Results Dir     : {base_path}")
    print("=" * 70)
    
    seed_everything(seed=args.seed_value)
    
    # Configure devices
    device_llm = "cuda" if torch.cuda.is_available() else "cpu"
    device_deberta = "cuda" if torch.cuda.is_available() else "cpu"
    
    auth_token = getattr(args, 'hf_token', None) or os.environ.get("HF_TOKEN", None)
    skip_gen = getattr(args, 'skip_generation', False)
    
    # Load DeBERTa
    print(f"Loading DeBERTa Model and Tokenizer: microsoft/{args.deberta_model}...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{args.deberta_model}", token=auth_token)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{args.deberta_model}",
        device_map="auto" if torch.cuda.is_available() else None,
        token=auth_token
    )
    if not torch.cuda.is_available():
        deberta_model = deberta_model.to(device_deberta)
        
    if not skip_gen:
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
                print("Configuring 4-bit (NF4) quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quant_mode == '8bit':
                print("Configuring 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
        if torch.cuda.is_available():
            llm_model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                token=auth_token,
                trust_remote_code=trust_remote,
                attn_implementation=attn_impl,
                device_map="auto"
            )
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                token=auth_token,
                trust_remote_code=trust_remote
            ).to(device_llm)
            
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
            torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
        ).squeeze().detach()
        
    # Load evaluation metrics
    print("Loading evaluation metrics (ROUGE, Exact Match, BLEURT-base-128)...")
    exact_match_metric = evaluate.load("exact_match")
    rouge = evaluate.load("rouge")
    bleurt = evaluate.load("bleurt", "bleurt-base-128")
    
    # Load ASQA dataset
    val_samples = load_asqa_samples(split=args.split, num_samples=args.num_samples)
    total_dataset_len = len(val_samples)
    
    start_idx = getattr(args, 'start_idx', 0)
    num_samples = getattr(args, 'num_samples', -1)
    if num_samples is None or num_samples <= 0:
        end_idx = total_dataset_len
    else:
        end_idx = min(start_idx + num_samples, total_dataset_len)
        
    failed_indices = []
    
    if not skip_gen:
        # Check existing result files for safe resumption
        existing_files = glob.glob(os.path.join(base_path, 'results_dict_*.pkl*'))
        existing_indices = []
        for f in existing_files:
            fname = os.path.basename(f)
            if fname.startswith('results_dict_semantic_matrices_'):
                continue
            clean_name = fname.replace('results_dict_', '').replace('.pkl.gz', '').replace('.pkl', '')
            try:
                existing_indices.append(int(clean_name))
            except ValueError:
                continue
                
        max_existing_idx = -1
        if existing_indices:
            max_existing_idx = max(existing_indices)
            
        resume_idx = start_idx
        if max_existing_idx >= start_idx and not getattr(args, 'force_overwrite', False):
            # Verify the last file isn't corrupted
            last_corrupt = False
            for ext in ['.pkl.gz', '.pkl']:
                fpath = os.path.join(base_path, f'results_dict_{max_existing_idx}{ext}')
                if os.path.exists(fpath):
                    try:
                        if ext == '.pkl.gz':
                            with gzip.open(fpath, 'rb') as fp:
                                pickle.load(fp)
                        else:
                            with open(fpath, 'rb') as fp:
                                pickle.load(fp)
                    except Exception:
                        last_corrupt = True
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass
            if last_corrupt:
                resume_idx = max(start_idx, max_existing_idx)
                print(f"[RESUME] Detected corrupt last file at index {max_existing_idx}. Resuming from {resume_idx}.")
            else:
                resume_idx = max(start_idx, max_existing_idx + 1)
                print(f"[RESUME] Resuming generation loop from index {resume_idx} (completed through {max_existing_idx}).")
                
        print(f"\nProcessing ASQA samples [{resume_idx} to {end_idx - 1}] (Total: {end_idx - resume_idx})...\n")
        
        for idx in tqdm(range(resume_idx, end_idx), desc="Running ASQA experiment generations"):
            try:
                sample = val_samples[idx]
                question = sample['question']
                best_answer = sample['best_answer']
                correct_answers = sample['correct_answers']
                incorrect_answers = sample['incorrect_answers']
                row_id = sample['id']
                qa_pairs = sample['qa_pairs']
                annotations = sample['annotations']
                
                # Chat template prompt: single, continuous, fluent paragraph without lists, headers, or line breaks
                messages = [
                    {"role": "system", "content": "Provide a single, continuous, fluent paragraph that directly synthesizes the different answers or interpretations to the question. Start immediately with the answer. Do not use bullet points, numbered lists, markdown headers, line breaks, or introductory filler."},
                    {"role": "user", "content": question}
                ]
                try:
                    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt_text = f"Provide a single, continuous, fluent paragraph that directly synthesizes the different answers or interpretations to the following question. Start immediately with the answer without bullet points, lists, headers, or line breaks:\nQuestion: {question}\nAnswer:"
                    
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
                
                # 2. Compute Correctness Metric (ROUGE, EM, and candidate BLEURT)
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
                    'sample_id': row_id,
                    'best_answer': best_answer,
                    'correct_answers': correct_answers,
                    'incorrect_answers': incorrect_answers,
                    'qa_pairs': qa_pairs,
                    'annotations': annotations,
                    'correctness_dict': correctness_dict,
                    'sdlg': {
                        'generations': [most_likely_generation],
                        'likelihoods': [most_likely_generation_likelihoods],
                        'epistem_entropies': []
                    }
                }
                
                # 4. Generate Alternative Diverse Responses (SDLG)
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
                if not getattr(args, 'store_logits', True):
                    for key in ['sdlg']:
                        if key in results_dict:
                            for lik in results_dict[key].get('likelihoods', []):
                                if 'generation_logits' in lik:
                                    lik['generation_logits'] = []
                                if 'cleaned_generation_logits' in lik:
                                    lik['cleaned_generation_logits'] = []
                
                # Save results_dict
                if getattr(args, 'compress_pkl', True):
                    out_path = os.path.join(base_path, f"results_dict_{idx}.pkl.gz")
                    with gzip.open(out_path, 'wb') as fp:
                        pickle.dump(results_dict, fp)
                else:
                    out_path = os.path.join(base_path, f"results_dict_{idx}.pkl")
                    with open(out_path, 'wb') as fp:
                        pickle.dump(results_dict, fp)

                # Real-time console logging showing SDLG alternatives
                rougel_val = correctness_dict.get('rougeL', 0.0)
                em_val = correctness_dict.get('exact_match', 0.0)
                
                tqdm.write(f"\n[Sample #{idx} (ID:{row_id})] Question: {question}")
                tqdm.write(f"  Greedy Answer : '{most_likely_generation['generation_text'][0]}'")
                ref_preview = correct_answers[:2] if len(correct_answers) > 2 else correct_answers
                tqdm.write(f"  Reference Ans : {ref_preview}")
                tqdm.write(f"  Correctness   : ROUGE-L = {rougel_val:.4f} | EM = {em_val:.1f} (BLEURT computed in batched pass at end)")
                tqdm.write(f"  SDLG Alts ({len(results_dict['sdlg']['generations']) - 1}):")
                for g_idx, gen in enumerate(results_dict['sdlg']['generations']):
                    if g_idx > 0:
                        tqdm.write(f"   [{g_idx}] '{gen['generation_text'][0]}'")
                tqdm.write("-" * 60)
                
                # Free VRAM to prevent leaks and fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                        
            except Exception as e:
                failed_indices.append((idx, str(e)))
                print(f"[ERROR] Error processing sample index {idx}: {e}")
                traceback.print_exc()
                if "CUDA error" in str(e) or "illegal memory access" in str(e):
                    print("[FATAL] CUDA error encountered. Exiting script to prevent terminal flooding.")
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
    
    # 5. Automatically compute NLI semantic equivalence and sequence-level importance scores
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
            print(f"[ERROR] Error computing semantic pairs: {e}")
            traceback.print_exc()

    # 6. Free GPU memory before batched BLEURT
    print("\nFreeing GPU memory before batched BLEURT...")
    if not skip_gen:
        if 'llm_model' in locals(): del llm_model
        if 'deberta_embeddings' in locals(): del deberta_embeddings
    if 'deberta_model' in locals(): del deberta_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. Fast Vectorized Batched BLEURT Scoring against multi-reference answers
    from batch_scoring import run_batched_bleurt_scoring
    bleurt_batch_sz = getattr(args, 'bleurt_batch_size', 64)
    run_batched_bleurt_scoring(base_path=base_path, batch_size=bleurt_batch_sz)

    # 8. Auto-Trigger Batch Clustering & 3xN Pickle Export to result/entropies_unified/
    if getattr(args, 'auto_batch_compute', True):
        from batch_compute import run_batch_compute
        target_dir = getattr(args, 'entropies_dir', 'entropies_unified')
        run_batch_compute(args=args, base_path=base_path, target_entropies_dir=target_dir)

    status_str = "SUCCESS (with warnings)" if failed_indices else "SUCCESS"
    print("=" * 70)
    print(f"EXPERIMENT RUN FINISHED [{status_str}]:")
    print(f"  - Dataset                 : ASQA (din0s/asqa, {total_dataset_len} total instances)")
    print(f"  - Valid Processed Samples : {len(valid_results)}")
    print(f"  - Failed Samples in Loop  : {len(failed_indices)}")
    print(f"  - Raw Sample Directory    : {base_path}")
    print(f"  - Exported Entropies Dir  : {os.path.join(script_dir, 'result', getattr(args, 'entropies_dir', 'entropies_unified'))}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
