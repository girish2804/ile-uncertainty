import os
import sys
import glob
import gzip
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from all_layers_args import AllLayersArgs


def compute_all_layers_auroc_summary(run_id=None, correctness_threshold=-0.35):
    args = AllLayersArgs()
    if run_id is None:
        run_id = args.run_id

    base_path = os.path.join(script_dir, 'result', run_id)
    if not os.path.exists(base_path):
        print(f"[ERROR] Results directory not found: {base_path}")
        return

    files_gz = glob.glob(os.path.join(base_path, "results_dict_*.pkl.gz"))
    files_raw = glob.glob(os.path.join(base_path, "results_dict_*.pkl"))
    all_files = sorted(files_gz + files_raw, key=lambda p: int(os.path.basename(p).replace("results_dict_", "").split(".")[0]))

    if not all_files:
        print(f"[ERROR] No results_dict files found in {base_path}")
        return

    print(f"[INFO] Found {len(all_files)} result files in {base_path}. Parsing metrics...")

    samples_data = []
    y_true_labels = []

    for fpath in tqdm.tqdm(all_files, desc="Loading samples"):
        try:
            if fpath.endswith('.gz'):
                with gzip.open(fpath, 'rb') as f:
                    res = pkl.load(f)
            else:
                with open(fpath, 'rb') as f:
                    res = pkl.load(f)

            c_dict = res.get('correctness_dict', {})
            bleurt_score = c_dict.get('bleurt', c_dict.get('bleurt_score', 0.0))
            is_correct = 1 if bleurt_score >= correctness_threshold else 0
            y_true_labels.append(is_correct)

            epistemic_list = res.get('sdlg', {}).get('epistem_entropies', [])
            if epistemic_list and isinstance(epistemic_list, list) and len(epistemic_list) > 0:
                # Average metrics across evaluated substitution positions
                all_keys = epistemic_list[0].keys()
                avg_sample_dict = {}
                for k in all_keys:
                    vals = [step[k] for step in epistemic_list if k in step and isinstance(step[k], (int, float))]
                    if vals:
                        avg_sample_dict[k] = sum(vals) / len(vals)
                samples_data.append(avg_sample_dict)
            else:
                samples_data.append({})

        except Exception as e:
            print(f"[WARNING] Could not read {fpath}: {e}")

    df = pd.DataFrame(samples_data)
    y_true = np.array(y_true_labels)
    y_error = 1 - y_true

    print("\n" + "=" * 80)
    print(f"ALL-LAYERS AUROC EVALUATION SUMMARY (Threshold = {correctness_threshold})")
    print(f"Total Samples: {len(y_true)} | Accuracy: {np.mean(y_true)*100:.2f}% | Errors: {np.sum(y_error)}")
    print("=" * 80)

    if np.sum(y_error) == 0 or np.sum(y_error) == len(y_error):
        print("[ERROR] Label distribution collapsed. Cannot compute AUROC.")
        return

    # 1. Depth comparison table across all top_k layers
    depth_cols_em = sorted([c for c in df.columns if c.startswith('entropy_mixture_top_')], key=lambda x: int(x.split('_')[-1]))
    depth_cols_mi = sorted([c for c in df.columns if c.startswith('mutual_info_top_') and not c.endswith('_skip2')], key=lambda x: int(x.split('_')[-1]))
    depth_cols_gmi = sorted([c for c in df.columns if c.startswith('gated_mutual_info_top_') and not c.endswith('_skip2')], key=lambda x: int(x.split('_')[-1]))
    depth_cols_mi_skip = sorted([c for c in df.columns if c.startswith('mutual_info_top_') and c.endswith('_skip2')], key=lambda x: int(x.split('_')[-2]))
    depth_cols_gmi_skip = sorted([c for c in df.columns if c.startswith('gated_mutual_info_top_') and c.endswith('_skip2')], key=lambda x: int(x.split('_')[-2]))

    records = []
    max_k = max([int(c.split('_')[-1]) for c in depth_cols_em] + [0])

    for k in range(1, max_k + 1):
        em_col = f'entropy_mixture_top_{k}'
        mi_col = f'mutual_info_top_{k}'
        gmi_col = f'gated_mutual_info_top_{k}'
        mi_skip_col = f'mutual_info_top_{k}_skip2'
        gmi_skip_col = f'gated_mutual_info_top_{k}_skip2'

        def _get_auroc(col):
            if col in df.columns and df[col].notna().any():
                try:
                    return roc_auc_score(y_error, df[col].values)
                except Exception:
                    return np.nan
            return np.nan

        rec = {
            'Depth (K)': k,
            'AUROC_H_mix (Aleatoric)': _get_auroc(em_col),
            'AUROC_Original_MI': _get_auroc(mi_col),
            'AUROC_Gated_MI (Ours)': _get_auroc(gmi_col),
            'AUROC_MI_Skip2': _get_auroc(mi_skip_col),
            'AUROC_Gated_MI_Skip2': _get_auroc(gmi_skip_col)
        }
        records.append(rec)

    depth_df = pd.DataFrame(records)
    print("\n--- Per-Layer Ensembling Depth Sweep Table ---")
    print(depth_df.to_string(index=False))

    # 2. Depth 75% Normalized Benchmark
    print("\n" + "-" * 80)
    print("DEPTH-NORMALIZED 75% METRICS:")
    for col_name in ['entropy_mixture_depth_75', 'mutual_info_depth_75', 'gated_mutual_info_depth_75',
                     'mutual_info_depth_75_skip2', 'gated_mutual_info_depth_75_skip2']:
        if col_name in df.columns:
            try:
                score = roc_auc_score(y_error, df[col_name].values)
                print(f"  {col_name:<35}: AUROC = {score:.4f}")
            except Exception:
                pass
    print("-" * 80 + "\n")

    # Save summary CSV
    out_csv = os.path.join(base_path, 'all_layers_auroc_summary.csv')
    depth_df.to_csv(out_csv, index=False)
    print(f"[INFO] Summary CSV saved to: {out_csv}")


if __name__ == '__main__':
    compute_all_layers_auroc_summary()
