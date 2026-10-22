import os
import pandas as pd
import torch
import pickle as pkl
import json
import torch.nn as nn
import numpy as np
import tqdm
import traceback
import pdb
import csv
from sklearn.metrics import roc_auc_score
import gc  # Add garbage collection

# Assuming 'utils' and 'args' are in the same directory or accessible
from utils import compute_semantic_clusters, compute_semantic_entropy_new, prepare_results, compute_semantic_entropy_var
from args import Args

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
args = Args()
auth_token = os.environ.get('HF_TOKEN', None)
device_llm = 'cuda:0'
device_deberta = 'cuda:0'
removed_sample_ids = []
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
base_path = os.path.join(repo_root, "result", "truth-mist_var")
base_path_cluster = os.path.join(repo_root, "result", "truth-mist_var")
base_path_final = repo_root
num_instances = 817 # dataset instances
model_type = args.deberta_model
correctness_metric = ["rougeL", "bleurt", "rouge1"][1]
num_total_gens = 10
correctness_threshold_list = [-0.05, -0.1, -0.15, -0.2, -0.3, -0.4, -0.5, 0, 0.03, 0.07, 0.1, .13, .16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
run_keys = ['sdlg'] #, 'baseline']
compute_cleaned = args.compute_cleaned

# Keys for the baseline methods
auroc_keys_baseline = ["normalised_semantic_entropy", "unnormalised_semantic_entropy"] 

# Dynamically create the auroc_keys list for 'ours' method
auroc_keys = ["normalised_semantic_entropy", "unnormalised_semantic_entropy"]
var_n = 31

# Assuming var_n elements for entropy_mixture, from index 0 to 30
for i in range(var_n): 
    auroc_keys.append(f"entropy_mixture_{i}")

# Dynamically create the CSV header
csv_header = [
    "run_key",
    "num_gens",
    "correctness_threshold",
    "auroc_norm_sem_ent",
    "auroc_unnorm_sem_ent",
    "auroc_ours_norm_sem_ent",
    "auroc_ours_unnorm_sem_ent",
]
for i in range(var_n):
    csv_header.append(f'{i}_ent_mix_auroc')
csv_header.extend([
    "auroc_kuhn_norm_ent",
    "auroc_kuhn_unnorm_ent",
])

csv_filepath = os.path.join(base_path_final, 'auroc_var_truth_mist.csv')
if not os.path.exists(csv_filepath):
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

for run_key in run_keys:
    # Initialize storage for all results
    all_bleurt_scores = []
    all_semantic_entropies_ours_final = []

    batch_size = 100
    print(f"Processing {num_instances} instances in batches of {batch_size}")

    for batch_start in tqdm.tqdm(range(0, num_instances, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, num_instances)
        print(f"Processing batch {batch_start}-{batch_end-1}")

        list_results_dict, list_bleurt, _ = prepare_results(num_samples=batch_end,
                                                            run_key=run_key,
                                                            metric=correctness_metric,
                                                            start_sample_id=batch_start,
                                                            base_path=base_path_cluster)
        
        # for results_dict in list_results_dict:
        #     if 'epistem_entropies' in results_dict[run_key]:
        #         for y in results_dict[run_key]['epistem_entropies']:
        #             for x in y:
                        # x.pop('unfiltered_premature_logits')
                        # x.pop('unfiltered_final_logits')
                        # x.pop('tokens_kept_count')
                        # x.pop('tokens_kept_percentage')
                        # x.pop('entropy_mixture_min')
                        # x.pop('entropy_mixture_max')
                        # x.pop('entropy_final')

        batch_bleurt_scores = []
        batch_semantic_entropies_ours = []

        for i, results_dict in enumerate(list_results_dict):
            try:
                # --- MODIFICATION START ---
                # Reverting to the original, robust masking logic
                num_gens = num_total_gens
                
                # This correctly creates a mask with the same length as the number of available generations
                if num_gens < len(results_dict[run_key]['generations']):
                    boolean_mask = [True] * num_gens + [False] * (len(results_dict[run_key]['generations']) - num_gens)
                else:
                    boolean_mask = [True] * len(results_dict[run_key]['generations'])
                
                boolean_mask = torch.tensor(boolean_mask)
                # --- MODIFICATION END ---
                
                # Now the boolean_mask will always have the correct dimension for indexing
                semantic_pairs = results_dict[run_key][f'semantic_pairs_{model_type}']['semantic_pairs'][boolean_mask, :][:, boolean_mask]

                all_considered_generations = [gen for m, gen in enumerate(results_dict[run_key]['generations']) if boolean_mask[m]]
                all_considered_likelihoods = [lik for m, lik in enumerate(results_dict[run_key]['likelihoods']) if boolean_mask[m]]
                
                my_weights = torch.nn.functional.normalize(results_dict[run_key]['seq_level_impo'][boolean_mask].to(torch.float32), p=1, dim=0)
                
                semantic_difference = compute_semantic_clusters(generations=all_considered_generations,
                                                                cleaned_semantic_pairs=cleaned_semantic_pairs if compute_cleaned else None, 
                                                                semantic_pairs=semantic_pairs, 
                                                                compute_cleaned=compute_cleaned)

                # Compute the 'ours' version of entropy
                semantic_entropy_ours = compute_semantic_entropy_var(
                    weights=my_weights,
                    mc_estimate_over_clusters=False,
                    neg_log_likelihoods=all_considered_likelihoods, 
                    semantic_difference=semantic_difference,
                    epistem_entropy_list=results_dict[run_key]['epistem_entropies'],
                    compute_cleaned=compute_cleaned
                )
                # print(semantic_entropy_ours)

                batch_bleurt_scores.append(list_bleurt[i])
                batch_semantic_entropies_ours.append(semantic_entropy_ours)

            except Exception as e:
                print(f"Error in batch {batch_start}, instance {i}: {str(e)}. Skipping sample.")
                traceback.print_exc()
                continue
        
        all_bleurt_scores.extend(batch_bleurt_scores)
        all_semantic_entropies_ours_final.extend(batch_semantic_entropies_ours)
        
        del list_results_dict, list_bleurt, batch_bleurt_scores, batch_semantic_entropies_ours
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\nStarting AUROC calculations with {len(all_bleurt_scores)} total samples...")
    
    for correctness_threshold in correctness_threshold_list:
        print(f'Processing threshold: {correctness_threshold}')
        try:
            if not all_bleurt_scores:
                print("  - Skipping threshold, no scores available.")
                continue

            list_correct_labels = (torch.tensor(all_bleurt_scores) < correctness_threshold)
            
            aurocs_ours = {}
            
            if all_semantic_entropies_ours_final:
                for key in auroc_keys:  
                    aurocs_ours[key] = roc_auc_score(list_correct_labels, [d.get(key, 0.0) for d in all_semantic_entropies_ours_final])
           
            row_to_write = [
                run_key,
                num_total_gens,
                correctness_threshold,
                11.0, # Placeholder for aurocs (norm)
                11.0, # Placeholder for aurocs (unnorm)
                aurocs_ours.get("normalised_semantic_entropy", 11.0),
                aurocs_ours.get("unnormalised_semantic_entropy", 11.0),
            ]
            
            for i in range(31):
                key = f'entropy_mixture_{i}'
                row_to_write.append(aurocs_ours.get(key, 11.0))
                
            row_to_write.extend([
                11.0, # Placeholder for aurocs_kuhn (norm)
                11.0, # Placeholder for aurocs_kuhn (unnorm)
            ])

            with open(csv_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_to_write)
                    
        except Exception as e:
            print(f'Error at threshold {correctness_threshold}: {str(e)}')
            traceback.print_exc()

    del all_bleurt_scores, all_semantic_entropies_ours_final
    gc.collect()
        
    print(f"Completed processing for run_key: {run_key}")

print("All processing completed!")