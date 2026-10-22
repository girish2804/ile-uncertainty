import os
import glob
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve


def get_curves_and_metrics(df, uncertainty_col):
    correct = df['Is_Correct'].values
    u_val = df[uncertainty_col].values
    N = len(correct)
    a0 = correct.mean()
    
    # 1. Compute AUROC
    try:
        auroc = roc_auc_score(1 - correct, u_val)
    except Exception:
        auroc = 0.5
        
    # 2. Compute PRR (Prediction Rejection Ratio)
    sorted_indices = df.sort_values(by=uncertainty_col, ascending=True).index.values
    coverages = np.linspace(0.01, 1.0, 100)
    
    accuracies_method = []
    for cov in coverages:
        num_keep = max(1, int(N * cov))
        accuracies_method.append(correct[sorted_indices[:num_keep]].mean())
        
    accuracies_oracle = []
    num_correct = correct.sum()
    for cov in coverages:
        num_keep = max(1, int(N * cov))
        kept_correct = min(num_keep, num_correct)
        accuracies_oracle.append(kept_correct / num_keep)
        
    def trapz(y, x):
        return sum(0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i]) for i in range(len(y)-1))
        
    area_method = trapz([acc - a0 for acc in accuracies_method], coverages)
    area_oracle = trapz([acc - a0 for acc in accuracies_oracle], coverages)
    prr = area_method / (area_oracle + 1e-10)
    
    return auroc, prr


def evaluate_ice_run(results_dir, corr_threshold=0.0):
    print("=" * 70)
    print(f"EVALUATING ICE RUN: {results_dir}")
    print("=" * 70)

    files = glob.glob(os.path.join(results_dir, "results_dict_*.pkl.gz")) + glob.glob(os.path.join(results_dir, "results_dict_*.pkl"))
    if not files:
        print("No result files found in directory.")
        return

    data_list = []
    for filepath in files:
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            bleurt_score = data.get('correctness_dict', {}).get('bleurt', 0.0)
            ice = data.get('ice', {})
            
            data_list.append({
                'bleurt': bleurt_score,
                'epistemic': ice.get('epistemic_uncertainty', 0.0),
                'total': ice.get('total_uncertainty', 0.0),
                'aleatoric': ice.get('aleatoric_uncertainty', 0.0)
            })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    df = pd.DataFrame(data_list)
    if len(df) == 0:
        print("No valid sample data extracted.")
        return

    # Benchmark Invariance: Standard BLEURT threshold is 0.0 across all datasets
    CORR_THRESHOLD = corr_threshold
    df['Is_Correct'] = (df['bleurt'] >= CORR_THRESHOLD).astype(int)

    print(f"Total Evaluated Samples: {len(df)}")
    print(f"Accuracy (BLEURT >= {CORR_THRESHOLD:.2f}): {df['Is_Correct'].mean() * 100:.2f}%")
    print(f"Correct Count: {df['Is_Correct'].sum()} | Incorrect Count: {len(df) - df['Is_Correct'].sum()}")
    print("-" * 70)

    # First law of metric debugging: check label distribution
    if df['Is_Correct'].nunique() < 2:
        print(f"WARNING: Class distribution is degenerate (all {df['Is_Correct'].iloc[0]}). AUROC will collapse to 0.500.")
        print(f"BLEURT distribution: min={df['bleurt'].min():.4f}, mean={df['bleurt'].mean():.4f}, max={df['bleurt'].max():.4f}")

    # Compute metrics for each uncertainty type
    metrics_summary = []
    for col in ['epistemic', 'total', 'aleatoric']:
        auroc, prr = get_curves_and_metrics(df, col)
        metrics_summary.append({
            'Uncertainty Type': col.capitalize(),
            'AUROC (Error Detection)': f"{auroc:.4f}",
            'PRR (Rejection Performance)': f"{prr:.4f}"
        })

    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_markdown(index=False))
    print("=" * 70)


if __name__ == '__main__':
    import sys
    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    exp_ice_dir = os.path.abspath(os.path.join(analysis_dir, "..", "experiments", "ice"))
    if exp_ice_dir not in sys.path:
        sys.path.insert(0, exp_ice_dir)

    run_id = 'ice-scienceqa-qwen2.5-1.5b'
    try:
        from ice_args import ICEArgs
        args = ICEArgs()
        if hasattr(args, 'run_id'):
            run_id = args.run_id
    except Exception as e:
        print(f"Note: Could not import ICEArgs ({e}); using default run_id: {run_id}")

    target_results = None
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        target_results = sys.argv[1]
    else:
        candidates = [
            os.path.join(exp_ice_dir, "result", run_id),
            os.path.join(analysis_dir, "..", "experiments", "result", run_id),
            os.path.join(exp_ice_dir, "result", "ice-scienceqa-qwen2.5-1.5b"),
            os.path.join(exp_ice_dir, "result", "ice-truthfulqa-qwen2.5-1.5b")
        ]
        for c in candidates:
            if os.path.exists(c):
                target_results = c
                break

    if target_results and os.path.exists(target_results):
        evaluate_ice_run(target_results, corr_threshold=0.0)
    else:
        print(f"Could not find results directory for {run_id}.")
        print("Please provide results path as CLI argument or run an ICE experiment first.")
