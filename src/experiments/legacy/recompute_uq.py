import os
import sys
import pickle
import gzip
import torch
import glob
from tqdm import tqdm

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from args import Args
from disagree import calculate_all_disagreement_metrics
from sdlg_var import get_model_layer_count
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    args = Args()
    
    # 1. Configuration - Set this to match your run directory
    RUN_ID = 'truth-mist_var'
    base_path = os.path.join('result', RUN_ID)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Find all result files
    pkl_gz_files = glob.glob(os.path.join(base_path, 'results_dict_*.pkl.gz'))
    pkl_files = glob.glob(os.path.join(base_path, 'results_dict_*.pkl'))
    all_files = pkl_gz_files + pkl_files
    print(f"Found {len(all_files)} results files in {base_path}")
    
    if not all_files:
        print("No files found. Exiting.")
        return
        
    # 2. Load the model (required for the forward pass)
    # Reuses settings from var_layers_exp.py
    auth_token = os.environ.get('HF_TOKEN', None)
    args.llm_model = "mistralai/Mistral-7B-v0.1"  # Set to the model you ran
    
    print(f"Loading model: {args.llm_model}")
    model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, token=auth_token).to(device)
    model.eval()
    lm_head = model.get_output_embeddings()
    
    total_layers = get_model_layer_count(model)
    candidate_ranges = []
    for start_layer in range(total_layers - 2, -1, -1):
        candidate_ranges.append(list(range(start_layer, total_layers)))
        
    print(f"Recomputing UQ disagreement metrics...")
    
    # 3. Process each file
    for filepath in tqdm(all_files):
        is_gz = filepath.endswith('.gz')
        try:
            if is_gz:
                with gzip.open(filepath, 'rb') as f:
                    results_dict = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    results_dict = pickle.load(f)
                    
            input_ids = results_dict['input_ids'].to(device)
            
            # Update 'sdlg' run key
            run_key = 'sdlg'
            if run_key in results_dict and 'generations' in results_dict[run_key]:
                generations = results_dict[run_key]['generations']
                
                # Clear and recompute the ensembling metrics list
                results_dict[run_key]['epistem_entropies'] = []
                
                # Iterate over alternative generations (skip the first greedy one as UQ ensembling is calculated during substitutions)
                for gen_idx, gen in enumerate(generations):
                    if gen_idx == 0:
                        # For the first (greedy) generation, ensembling is empty
                        results_dict[run_key]['epistem_entropies'].append([])
                        continue
                        
                    # Reconstruct input sequence exactly as was fed to the model in sdlg_var.py
                    initial_generation_ids = gen['initial_generation_ids']
                    initial_gen_token_idx = gen['initial_gen_token_idx']
                    
                    if initial_gen_token_idx > 0:
                        new_input_ids = initial_generation_ids[:initial_gen_token_idx]
                        all_input_ids = torch.hstack([input_ids, new_input_ids.unsqueeze(0).to(device)])
                    else:
                        all_input_ids = input_ids
                        
                    # Single forward pass (extremely fast compared to autoregressive generation)
                    with torch.no_grad():
                        model_outputs = model(all_input_ids, return_dict=True, output_hidden_states=True)
                        final_logits = model_outputs.logits[:, -1, :].float()
                        
                        epistem_entropy = []
                        for candidate_premature_layers in candidate_ranges:
                            candidate_premature_logits = {}
                            for layer in candidate_premature_layers:
                                candidate_premature_logits[layer] = lm_head(
                                    model_outputs.hidden_states[layer][:, -1, :]
                                ).to(final_logits.device)
                                
                            # Calculate all metrics (including MI, dynamic mixture, and SVE)
                            metrics_to_compute = ['entropy_mixture', 'mutual_info', 
                                                 'entropy_mixture_dynamic', 'entropy_mixture_maxprob', 'entropy_mixture_decay', 'sve']
                            entropy_result = calculate_all_disagreement_metrics(
                                final_logits, 
                                candidate_premature_logits, 
                                calculate_metrics=metrics_to_compute, 
                                store_logits=False,
                                embeddings=model.get_output_embeddings().weight
                            )
                            epistem_entropy.append(entropy_result)
                            
                    results_dict[run_key]['epistem_entropies'].append(epistem_entropy)
            
            # Save the updated pickle file back
            if is_gz:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(results_dict, f)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(results_dict, f)
                    
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()

    print("Offline recomputation completed successfully!")

if __name__ == '__main__':
    main()
