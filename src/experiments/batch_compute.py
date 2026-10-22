import os
import sys
import pickle
import pickle as pkl
import gzip
import json
import csv
import traceback
import pdb
import glob
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score
import gc

# Add src/ to Python path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils import compute_semantic_clusters, compute_semantic_entropy, prepare_results
from args import Args


def run_batch_compute(args=None, base_path=None, target_entropies_dir=None, run_keys=None):
    """
    Executes semantic clustering, semantic entropy computation (ours, sdlg, kuhn),
    merges all in-situ epistemic disagreement metrics, and exports:
      - sdlg_label_<dataset>_<model>_new.pkl
      - sdlg_question_<dataset>_<model>_new.pkl
      - sdlg_entropies_<dataset>_<model>_new.pkl (3xN concatenation)
    directly into result/<target_entropies_dir>/ (default: entropies_unified).
    """
    if args is None:
        args = Args()

    result_parent_dir = os.path.join(script_dir, 'result')

    # 1. Configure dataset and model tags dynamically based on Args
    dataset_lower = getattr(args, 'dataset', 'truthful_qa').lower()
    if 'truth' in dataset_lower:
        data = 'truth'
        prefix = 'truthfulqa'
    elif 'open' in dataset_lower:
        data = 'open'
        prefix = 'openbookqa'
    elif 'science' in dataset_lower:
        data = 'scienceqa'
        prefix = 'scienceqa'
    elif 'fresh' in dataset_lower:
        data = 'fresh'
        prefix = 'freshqa'
    else:
        data = dataset_lower
        prefix = dataset_lower

    llm_model_str = str(getattr(args, 'llm_model', 'qwen')).split('/')[-1].lower()
    model_clean = llm_model_str.replace('-instruct', '').replace('-chat', '').replace('_', '-')
    if 'qwen2.5-1.5b' in model_clean or 'qwen-1.5b' in model_clean:
        modl = 'qwen-1.5b'
    else:
        modl = model_clean
    quant_suffix = f"_{getattr(args, 'quantization', '')}" if getattr(args, 'quantization', None) else ""
    modl = f"{modl}{quant_suffix}"

    # 2. Smart Multi-Candidate Results Folder Discovery (if base_path not specified)
    if base_path is None:
        candidate_folders = []
        if hasattr(args, 'run_id') and args.run_id:
            candidate_folders.append(args.run_id)
        if hasattr(args, 'dynamic_run_id') and args.dynamic_run_id:
            candidate_folders.append(args.dynamic_run_id)

        candidate_folders.extend([
            f"{prefix}_{model_clean}",
            f"{prefix}-{model_clean}",
            f"{prefix}_{llm_model_str}",
            f"{prefix}-{llm_model_str}",
            f"{data}_{modl}",
            f"{data}-{modl}",
            f"{data}-{modl}-new",
        ])

        seen = set()
        unique_candidates = [x for x in candidate_folders if not (x in seen or seen.add(x))]

        for c_folder in unique_candidates:
            test_path = os.path.join(result_parent_dir, c_folder)
            if os.path.exists(test_path):
                files_found = glob.glob(os.path.join(test_path, "results_dict_*.pkl*"))
                if files_found:
                    base_path = test_path
                    break
                elif base_path is None:
                    base_path = test_path

        # Fuzzy fallback: Scan all directories in result/ for matches
        if base_path is None or not glob.glob(os.path.join(base_path, "results_dict_*.pkl*")):
            if os.path.exists(result_parent_dir):
                all_dirs = [d for d in os.listdir(result_parent_dir) if os.path.isdir(os.path.join(result_parent_dir, d))]
                for d in all_dirs:
                    d_lower = d.lower()
                    if (prefix in d_lower or data in d_lower) and (model_clean in d_lower or modl in d_lower):
                        test_path = os.path.join(result_parent_dir, d)
                        files_found = glob.glob(os.path.join(test_path, "results_dict_*.pkl*"))
                        if files_found:
                            base_path = test_path
                            print(f"[AUTO-DETECT] Found matching results directory with {len(files_found)} files: {d}")
                            break

        if base_path is None:
            base_path = os.path.join(result_parent_dir, unique_candidates[0])

    base_path_cluster = base_path
    base_path_final = script_dir

    # 3. Configure target entropies directory (default: entropies_unified)
    if target_entropies_dir is None:
        target_entropies_dir = getattr(args, 'entropies_dir', 'entropies_unified')
    base_path_entropies = os.path.join(script_dir, 'result', target_entropies_dir)
    os.makedirs(base_path_entropies, exist_ok=True)

    print("=" * 60)
    print("BATCH COMPUTE & SEMANTIC ENTROPY AGGREGATION")
    print(f"Active LLM Model:           {args.llm_model}")
    print(f"Active Dataset:             {args.dataset} (Tag: {data})")
    print(f"Results Source Directory:   {base_path}")
    print(f"Target Entropies Directory: {base_path_entropies}")
    print("=" * 60)

    # 4. Detect sample files
    existing_pkl_gz = glob.glob(os.path.join(base_path, "results_dict_*.pkl.gz"))
    existing_pkl = glob.glob(os.path.join(base_path, "results_dict_*.pkl"))
    existing_files = [
        f for f in (existing_pkl_gz + existing_pkl)
        if not os.path.basename(f).startswith("results_dict_semantic_matrices_")
    ]

    sample_indices = []
    for f in existing_files:
        basename = os.path.basename(f)
        idx_str = basename.replace("results_dict_", "").split(".")[0]
        try:
            sample_indices.append(int(idx_str))
        except ValueError:
            pass
    sample_indices = sorted(list(set(sample_indices)))

    if sample_indices:
        num_instances = len(sample_indices)
        print(f"Detected {num_instances} valid result files across indices [{sample_indices[0]} .. {sample_indices[-1]}].")
    else:
        if data in ['open', 'openbook']:
            num_instances = 500
        elif data in ['truth', 'truthful']:
            num_instances = 817
        elif 'science' in data:
            num_instances = 2214
        else:
            num_instances = 2224
        sample_indices = list(range(num_instances))
        print(f"[WARNING] No result files found in {base_path}. Using fallback default range: {num_instances}")

    model_type = getattr(args, 'deberta_model', 'deberta-large-mnli')

    # Detect correctness metric
    first_keys = []
    first_file = existing_files[0] if existing_files else None
    if first_file and os.path.exists(first_file):
        try:
            if first_file.endswith('.gz'):
                with gzip.open(first_file, 'rb') as f:
                    first_results = pickle.load(f)
            else:
                with open(first_file, 'rb') as f:
                    first_results = pickle.load(f)
            if first_results and 'correctness_dict' in first_results:
                first_keys = list(first_results['correctness_dict'].keys())
        except Exception as e:
            print(f"[DEBUG] Could not inspect first file: {e}")

    if first_keys:
        if 'bleurt' in first_keys:
            correctness_metric = 'bleurt'
        elif 'rougeL' in first_keys:
            correctness_metric = 'rougeL'
        elif 'rouge1' in first_keys:
            correctness_metric = 'rouge1'
        elif 'exact_match' in first_keys:
            correctness_metric = 'exact_match'
        else:
            correctness_metric = first_keys[0]
    else:
        correctness_metric = 'bleurt'

    print(f"Using correctness metric: {correctness_metric}")

    num_total_gens = getattr(args, 'num_total_generations', 10)
    auroc_keys_baseline = ["normalised_semantic_entropy", "unnormalised_semantic_entropy"]
    auroc_keys = ["normalised_semantic_entropy", "unnormalised_semantic_entropy", 'entropy_final',
                  'jsd', 'kl', 'entropy_mixture', 'mutual_info', 'epkl', 'cv', 'bhattacharyya', 'tvd']
    correctness_threshold_list = [-0.05, -0.1, -0.15, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, 0, 0.03, 0.07, 0.1, .13, .16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if run_keys is None:
        run_keys = ['sdlg']
    compute_cleaned = getattr(args, 'compute_cleaned', False)

    csv_header = [
        "run_key",
        "num_gens",
        "correctness_threshold",
        "auroc_norm_sem_ent",
        "auroc_unnorm_sem_ent",
        "auroc_ours_norm_sem_ent",
        "auroc_ours_unnorm_sem_ent", 'entropy_final',
        'jsd', 'kl', 'entropy_mixture', 'mutual_info', 'epkl', 'cv', 'bhattacharyya', 'tvd',
        "auroc_kuhn_norm_ent",
        "auroc_kuhn_unnorm_ent",
    ]

    csv_filepath = os.path.join(base_path_final, f'auroc2_{data}_{modl}_new.csv')
    if not os.path.exists(csv_filepath):
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    for run_key in run_keys:
        batch_size = 100
        all_bleurt_scores = []
        all_batch_questions = []
        all_semantic_entropies_final = []
        all_semantic_entropies_kuhn_final = []
        all_semantic_entropies_ours_final = []

        print(f"Processing {len(sample_indices)} instances in batches of {batch_size}")

        # PHASE 1: BATCH PROCESSING
        for chunk_start in tqdm.tqdm(range(0, len(sample_indices), batch_size), desc="Processing batches"):
            current_chunk_indices = sample_indices[chunk_start : chunk_start + batch_size]
            current_batch_size = len(current_chunk_indices)

            list_results_dict, list_bleurt, dataset_size = prepare_results(
                num_samples=len(current_chunk_indices),
                run_key=run_key,
                metric=correctness_metric,
                start_sample_id=0,
                base_path=base_path_cluster,
                sample_indices=current_chunk_indices
            )

            batch_bleurt_scores = []
            batch_questions = []
            batch_semantic_entropies, batch_semantic_entropies_kuhn = [], []
            batch_semantic_entropies_ours = []

            for i, r_dict in enumerate(list_results_dict):
                try:
                    target_k = run_key if run_key in r_dict else ('sdlg' if 'sdlg' in r_dict else ('ours' if 'ours' in r_dict else 'baseline'))
                    gens_len = len(r_dict[target_k]['generations'])
                    if num_total_gens < gens_len:
                        boolean_mask = [True] * num_total_gens + [False] * (gens_len - num_total_gens)
                    else:
                        boolean_mask = [True] * gens_len

                    boolean_mask = torch.tensor(boolean_mask)

                    if target_k in ("sdlg", 'dola_sdlg'):
                        token_liks = []
                        for gen in r_dict[target_k]['generations'][1:]:
                            t_lik = gen.get('token_likelihood', 0.5)
                            if isinstance(t_lik, torch.Tensor):
                                t_lik = t_lik.item()
                            token_liks.append(min(max(float(t_lik), 1e-4), 0.9999))
                        mask = torch.tensor([1.0] + token_liks)
                    else:
                        mask = boolean_mask.float()

                    all_considered_generations, all_considered_likelihoods = [], []
                    for m, included in enumerate(boolean_mask):
                        if included:
                            all_considered_generations.append(r_dict[target_k]['generations'][m])
                            all_considered_likelihoods.append(r_dict[target_k]['likelihoods'][m])

                    # Find semantic pairs
                    pairs_key = f'semantic_pairs_{model_type}'
                    if pairs_key not in r_dict[target_k]:
                        for k in r_dict[target_k].keys():
                            if k.startswith('semantic_pairs'):
                                pairs_key = k
                                break

                    if pairs_key in r_dict[target_k] and r_dict[target_k][pairs_key]['semantic_pairs'].shape[0] > 1:
                        raw_pairs = r_dict[target_k][pairs_key]['semantic_pairs']
                        if raw_pairs.shape[0] >= len(boolean_mask):
                            semantic_pairs = raw_pairs[boolean_mask, :][:, boolean_mask]
                        else:
                            semantic_pairs = raw_pairs
                        cleaned_semantic_pairs = r_dict[target_k][pairs_key].get('cleaned_semantic_pairs', semantic_pairs)
                    else:
                        semantic_pairs = np.eye(len(all_considered_generations), dtype=bool)
                        cleaned_semantic_pairs = semantic_pairs

                    semantic_pairs = semantic_pairs & semantic_pairs.T
                    if compute_cleaned:
                        cleaned_semantic_pairs = cleaned_semantic_pairs & cleaned_semantic_pairs.T

                    semantic_difference = compute_semantic_clusters(
                        generations=all_considered_generations,
                        cleaned_semantic_pairs=cleaned_semantic_pairs if compute_cleaned else None,
                        semantic_pairs=semantic_pairs,
                        compute_cleaned=compute_cleaned
                    )

                    weights = boolean_mask[boolean_mask].to(torch.float32) if target_k == "baseline" else torch.nn.functional.normalize(mask[boolean_mask].to(torch.float32), p=1, dim=0)

                    if 'seq_level_impo' in r_dict[target_k] and r_dict[target_k]['seq_level_impo'] is not None:
                        seq_imp = r_dict[target_k]['seq_level_impo']
                        if isinstance(seq_imp, torch.Tensor) and len(seq_imp) >= len(boolean_mask):
                            my_weights = torch.nn.functional.normalize(seq_imp[boolean_mask].to(torch.float32), p=1, dim=0)
                        else:
                            my_weights = weights
                    else:
                        my_weights = weights

                    epistem_list = r_dict[target_k].get('epistem_entropies', [])

                    # 1. Standard unweighted SDLG semantic entropy
                    semantic_entropy_1 = compute_semantic_entropy(
                        weights=weights,
                        mc_estimate_over_clusters=False,
                        neg_log_likelihoods=all_considered_likelihoods,
                        semantic_difference=semantic_difference,
                        epistem_entropy_list=epistem_list,
                        compute_cleaned=compute_cleaned
                    )

                    # 2. Kuhn semantic entropy baseline
                    semantic_entropy_kuhn = compute_semantic_entropy(
                        weights=weights,
                        mc_estimate_over_clusters=True,
                        neg_log_likelihoods=all_considered_likelihoods,
                        semantic_difference=semantic_difference,
                        epistem_entropy_list=epistem_list,
                        compute_cleaned=compute_cleaned
                    )

                    # 3. Our weighted semantic entropy + all in-situ epistemic disagreement metrics
                    semantic_entropy_ours = compute_semantic_entropy(
                        weights=my_weights,
                        mc_estimate_over_clusters=False,
                        neg_log_likelihoods=all_considered_likelihoods,
                        semantic_difference=semantic_difference,
                        epistem_entropy_list=epistem_list,
                        compute_cleaned=compute_cleaned
                    )

                    if i < len(list_bleurt):
                        batch_bleurt_scores.append(list_bleurt[i])
                    else:
                        batch_bleurt_scores.append(0.0)

                    batch_questions.append(r_dict.get('question', ''))
                    batch_semantic_entropies.append(semantic_entropy_1)
                    batch_semantic_entropies_kuhn.append(semantic_entropy_kuhn)
                    batch_semantic_entropies_ours.append(semantic_entropy_ours)

                except Exception as e:
                    print(f"[DEBUG] Error processing instance #{i} in chunk {chunk_start}: {e}")
                    continue

            all_bleurt_scores.extend(batch_bleurt_scores)
            all_batch_questions.extend(batch_questions)
            all_semantic_entropies_final.extend(batch_semantic_entropies)
            all_semantic_entropies_kuhn_final.extend(batch_semantic_entropies_kuhn)
            all_semantic_entropies_ours_final.extend(batch_semantic_entropies_ours)

            del list_results_dict, list_bleurt
            del batch_bleurt_scores, batch_semantic_entropies, batch_semantic_entropies_kuhn, batch_semantic_entropies_ours
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # PHASE 2: PERSIST 3 PICKLES INTO TARGET ENTROPIES DIRECTORY
        print(f"\nFinal verification: Total bleurt scores = {len(all_bleurt_scores)}, Expected = {num_instances}")
        print(f"Final verification: Total entropy samples = {len(all_semantic_entropies_final)}, Expected = {num_instances}")

        if len(all_bleurt_scores) == 0 or len(all_semantic_entropies_final) == 0:
            print(f"\n[ERROR] No valid samples were processed for {run_key}. Skipping pickle export to avoid corrupting files.")
            continue

        out_label_path = os.path.join(base_path_entropies, f'{run_key}_label_{data}_{modl}_new.pkl')
        out_question_path = os.path.join(base_path_entropies, f'{run_key}_question_{data}_{modl}_new.pkl')
        out_entropy_path = os.path.join(base_path_entropies, f'{run_key}_entropies_{data}_{modl}_new.pkl')

        with open(out_label_path, 'wb') as f:
            pkl.dump(all_bleurt_scores, f)
        with open(out_question_path, 'wb') as f:
            pkl.dump(all_batch_questions, f)

        # Standard 3xN chunk format: [Ours, SDLG, Kuhn]
        all_semantic_entropies_ = all_semantic_entropies_ours_final + all_semantic_entropies_final + all_semantic_entropies_kuhn_final
        with open(out_entropy_path, 'wb') as f:
            pkl.dump(all_semantic_entropies_, f)

        print("\n" + "=" * 80)
        print(f"EXPORT COMPLETE -> Saved 3 Notebook Pickles to: {base_path_entropies}/")
        print(f"  1. Labels File    : {os.path.basename(out_label_path)} ({len(all_bleurt_scores)} samples)")
        print(f"  2. Questions File : {os.path.basename(out_question_path)}")
        print(f"  3. Entropies File : {os.path.basename(out_entropy_path)} (3xN: {len(all_semantic_entropies_)} entries)")
        print("=" * 80)

        # PHASE 3: AUROC EVALUATION & CSV OUTPUT
        for r, correctness_threshold in enumerate(correctness_threshold_list):
            try:
                list_correct_labels = torch.logical_not((torch.tensor(all_bleurt_scores) >= correctness_threshold))
                aurocs, aurocs_kuhn = {}, {}
                aurocs_ours = {}

                for k, key in enumerate(auroc_keys_baseline):
                    aurocs[key] = roc_auc_score(list_correct_labels, [d[key] for d in all_semantic_entropies_final])
                    aurocs_kuhn[key] = roc_auc_score(list_correct_labels, [d[key] for d in all_semantic_entropies_kuhn_final])

                for k, key in enumerate(auroc_keys):
                    if all_semantic_entropies_ours_final and key in all_semantic_entropies_ours_final[0]:
                        aurocs_ours[key] = roc_auc_score(list_correct_labels, [d[key] for d in all_semantic_entropies_ours_final])
                    else:
                        aurocs_ours[key] = 11.0

                if all_semantic_entropies_ours_final:
                    all_keys = all_semantic_entropies_ours_final[0].keys()
                    extra_keys = [key for key in all_keys if key not in auroc_keys and key not in ["normalised_semantic_entropy", "unnormalised_semantic_entropy"]]
                    for key in sorted(extra_keys):
                        try:
                            score = roc_auc_score(list_correct_labels, [d[key] for d in all_semantic_entropies_ours_final])
                            aurocs_ours[key] = score
                        except Exception:
                            pass

                with open(csv_filepath, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        run_key,
                        num_total_gens,
                        correctness_threshold,
                        aurocs.get("normalised_semantic_entropy", 11.0),
                        aurocs.get("unnormalised_semantic_entropy", 11.0),
                        aurocs_ours.get("normalised_semantic_entropy", 11.0),
                        aurocs_ours.get("unnormalised_semantic_entropy", 11.0),
                        aurocs_ours.get("entropy_final", 11.0),
                        aurocs_ours.get("jsd", 11.0),
                        aurocs_ours.get("kl", 11.0),
                        aurocs_ours.get("entropy_mixture", 11.0),
                        aurocs_ours.get("mutual_info", 11.0),
                        aurocs_ours.get("epkl", 11.0),
                        aurocs_ours.get("cv", 11.0),
                        aurocs_ours.get("bhattacharyya", 11.0),
                        aurocs_ours.get("tvd", 11.0),
                        aurocs_kuhn.get("normalised_semantic_entropy", 11.0),
                        aurocs_kuhn.get("unnormalised_semantic_entropy", 11.0),
                    ])
            except Exception as e:
                pass

        del all_bleurt_scores, all_semantic_entropies_final, all_semantic_entropies_kuhn_final, all_semantic_entropies_ours_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[FINISHED] Batch compute completed successfully.")


if __name__ == '__main__':
    run_batch_compute()