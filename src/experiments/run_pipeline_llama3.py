import os
import sys
import torch
import yaml
import pickle
import gzip
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset

# Add src/ to Python path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layer_ensemble import generate_layer_ensemble_sequences, generate_semantically_diverse_output_sequences
from utils import generate_text, compute_likelihood
from args import Args

def run_actual_pipeline():
    print("=" * 60)
    print("RUNNING INTERMEDIATE LAYER ENSEMBLING (ILE) UNCERTAINTY PIPELINE ON ACTUAL DATASET")
    print("=" * 60)
    
    # 1. Initialize configuration arguments
    args = Args()
    args.token_prob_threshold = 0.005  # Probability threshold to consider candidate replacements
    args.num_total_generations = 10    # Number of alternative sequences to sample
    args.max_length_of_generated_sequence = 40
    args.alphas = (0.33, 0.33, 0.34)
    args.invalid_ids = []
    
    # 2. Configure Models
    args.llm_model = getattr(args, 'llm_model', "Qwen/Qwen2.5-1.5B-Instruct")
    auth_token = getattr(args, 'hf_token', os.environ.get("HF_TOKEN", None))
    
    print(f"Loading LLM Model and Tokenizer: {args.llm_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, token=auth_token)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
        device_map="auto",
        token=auth_token
    )
    
    device_llm = next(llm_model.parameters()).device
    print(f"LLM loaded on device: {device_llm}")
    
    # 3. Configure DeBERTa model (representation encoder for substitutions)
    args.deberta_model = "deberta-large-mnli"
    device_deberta = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading DeBERTa Model and Tokenizer: microsoft/{args.deberta_model}...")
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{args.deberta_model}")
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{args.deberta_model}", 
        device_map="auto" if torch.cuda.is_available() else None
    )
    if not torch.cuda.is_available():
        deberta_model = deberta_model.to(device_deberta)
        
    print("Extracting DeBERTa word embeddings matrix...")
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()
    
    # 4. Safely set up special tokens
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            llm_model.resize_token_embeddings(len(tokenizer))
            
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    # Set up robust EOS token IDs list to support chat/instruct model stop tokens (like Qwen's <|im_end|> and Llama's <|eot_id|>)
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
    for token_str in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    args.eos_token_ids = list(set(eos_ids))
    
    # 5. Load truthfulqa dataset (generation split)
    print("Loading truthfulqa (generation split) dataset...")
    dataset = load_dataset('truthfulqa/truthful_qa', 'generation')
    val_set = dataset['validation']
    print(f"Dataset loaded. Total validation samples: {len(val_set)}")
    
    # Let's run on 2 sample questions to demonstrate the pipeline
    num_samples_to_test = 2
    
    for idx in tqdm(range(num_samples_to_test), desc="Processing samples"):
        sample = val_set[idx]
        question = sample['question']
        best_answer = sample['best_answer']
        
        print("\n" + "=" * 50)
        print(f"Sample #{idx + 1}")
        print(f"Question: {question}")
        print(f"Ground Truth Best Answer: {best_answer}")
        print("=" * 50)
        
        # Use Hugging Face chat template for proper instruction formatting on chat/instruct models
        messages = [
            {"role": "system", "content": "Provide a direct, concise, single-sentence answer to the user's question. Do not repeat system instructions, do not repeat the prompt, do not list multiple options, and do not explain your answer."},
            {"role": "user", "content": question}
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = f"Answer the following question directly and concisely in a single sentence:\nQuestion: {question}\nAnswer:"
        
        # Tokenize prompt
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device_llm)
        prompt_ids = inputs["input_ids"][0].to('cpu')
        
        # 6. Generate the initial most likely response (greedy decoding)
        print("Generating initial response...")
        most_likely_generation = generate_text(
            args=args,
            model=llm_model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            len_prompt=len(prompt_ids),
            decoding_method="most_likely",
            device=device_llm
        )
        
        initial_text = most_likely_generation['generation_text'][0]
        print(f"Initial LLM Response: '{initial_text}'")
        
        # Compute likelihood/logits for the initial response
        initial_likelihoods = compute_likelihood(
            prompt=prompt_ids,
            generation=most_likely_generation,
            model=llm_model,
            device=device_llm,
            compute_cleaned=False,
            store_logits=True
        )
        
        # Prepare results dict
        results_dict = {
            'input_ids': inputs["input_ids"].to('cpu'),
            'question': question,
            'sdlg': {
                'generations': [most_likely_generation],
                'likelihoods': [initial_likelihoods],
                'epistem_entropies': []
            }
        }
        
        # 7. Generate Semantically Diverse Alternatives (SDLG Pipeline)
        print("Running Intermediate Layer Ensembling (ILE) evaluation...")
        results_dict = generate_semantically_diverse_output_sequences(
            results_dict=results_dict,
            deberta_model=deberta_model,
            deberta_tokenizer=deberta_tokenizer,
            device_deberta=device_deberta,
            deberta_embeddings=deberta_embeddings,
            model=llm_model,
            tokenizer=tokenizer,
            device_llm=device_llm,
            input_ids=inputs["input_ids"],
            prompt=prompt_ids,
            question=question,
            initial_generation=most_likely_generation,
            initial_likelihood=initial_likelihoods,
            key='sdlg',
            compute_epistemic=False,
            args=args
        )
        
        # 8. Print Results
        print("\n--- Pipeline Generation Summary ---")
        generations = results_dict['sdlg']['generations']
        print(f"Generated {len(generations) - 1} alternative semantic sequences:")
        
        for g_idx, gen in enumerate(generations):
            text = gen['generation_text'][0]
            
            if g_idx > 0:
                print(f" - '{text}'")
            else:
                print(f"Initial: '{text}'")
        
        # 9. Save Results
        base_path = 'results_pipeline'
        os.makedirs(base_path, exist_ok=True)
        # Strip logits if store_logits is disabled to save disk space
        if not getattr(args, 'store_logits', True):
            for key in ['clar_sdlg', 'sdlg', 'dola_sdlg', 'baseline']:
                if key in results_dict:
                    for lik in results_dict[key].get('likelihoods', []):
                        if 'generation_logits' in lik:
                            lik['generation_logits'] = []
                        if 'cleaned_generation_logits' in lik:
                            lik['cleaned_generation_logits'] = []
                            
        if getattr(args, 'compress_pkl', True):
            save_path = os.path.join(base_path, f'results_dict_{idx}.pkl.gz')
            with gzip.open(save_path, 'wb') as outfile:
                pickle.dump(results_dict, outfile)
            print(f"Saved results to {save_path}")
        else:
            save_path = os.path.join(base_path, f'results_dict_{idx}.pkl')
            with open(save_path, 'wb') as outfile:
                pickle.dump(results_dict, outfile)
            print(f"Saved results to {save_path}")

if __name__ == "__main__":
    try:
        run_actual_pipeline()
    except Exception as e:
        import traceback
        traceback.print_exc()
