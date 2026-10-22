#!/usr/bin/env python3
"""
multi-seed evaluation script with better result management.
Evaluates multiple seeds and provides organized output with timestamp and metadata.
"""

import pickle
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import bayesian_transformer

# # Add src path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# evaluation utils
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from run.utils.math import *
from run.utils.grader import *
from run.utils.qwen_math_parser import *

# extract answer from string
def extract(string):
    """Extract answer from string"""
    try:
        return strip_string(extract_answer(string, 'math'))
    except:
        return string

def extract_unc(output):
    """Extract uncertainty scores from output"""
    try:
        unc_list = output.uncertainties
        ll_list = output.logprobs
        
        au_list = []
        tu_list = []
        eu_list = []
        ll_list_total = []
        
        for d in unc_list:
            au_list.append(list(d.values())[0].aleatoric_uncertainty)
            tu_list.append(list(d.values())[0].total_uncertainty)
            eu_list.append(list(d.values())[0].epistemic_uncertainty)

        for d in ll_list:
            ll_list_total.append(list(d.values())[0].logprob)
        au = np.array(au_list).sum()
        tu = np.array(tu_list).sum()
        eu = np.array(eu_list).sum()
        ll = np.array(ll_list_total).mean()
        
        return {
            'll': ll, 'au': au, 'tu': tu, 'eu': eu,
        }
    except Exception as e:
        return None


def get_top_p_acc(df, p, col):
    """Get accuracy for top p% samples sorted by column"""
    df_sorted = df.sort_values(by=col, ascending=False)
    top_p = int(len(df_sorted) * p)
    if top_p == 0:
        top_p = 1
    return df_sorted.iloc[:top_p]['label'].mean()


def safe_auc(labels, scores):
    """Safely compute AUC score"""
    try:
        return roc_auc_score(labels, scores)
    except:
        return np.nan


def safe_auprc(labels, scores):
    """Safely compute AUPRC score"""
    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
        return auc(recall, precision)
    except:
        return np.nan


def create_results_directory(model, dataset, timestamp):
    """Create organized results directory structure"""
    results_dir = os.path.join(BASE_DIR, 'results', 'eval', f'{model}_{dataset}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_metadata(results_dir, args, seeds, successful_seeds):
    """Save evaluation metadata"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'model': args.model,
        'results_subdir': args.results_subdir,
        'seeds_requested': seeds,
        'seeds_successful': successful_seeds,
        'seeds_failed': [s for s in seeds if s not in successful_seeds],
        'success_rate': f"{len(successful_seeds)}/{len(seeds)}",
    }
    
    with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def evaluate_single_seed(dataset, model, seed, results_subdir):
    """Evaluate a single seed and return metrics"""
    # Find pickle files
    pattern = os.path.join(BASE_DIR, 'results', f'{model}_results_vllm_pg', dataset, f'seed{seed}', results_subdir, '*.pkl')
    files = glob(pattern)
    if not files:
        print(f"No files found for seed {seed}")
        return None
    
    # Load data
    data = []
    for file in files:
        try:
            with open(file, 'rb') as f:
                data += pickle.load(f)
        except Exception as e:
            print(f"Error loading {os.path.basename(file)}: {e}")
    
    if not data:
        print(f"No data loaded for seed {seed}")
        return None
    
    # Process data
    processed_data = []
    for sample in data:
        try:
            if isinstance(sample['result'], list):
                result = sample['result'][0]
            else:
                result = sample['result']

            answer = extract(result.outputs[0].text)
            ground_truth = sample['answer']
            
            # Compute label (correctness)
            if 'deepscaler' in dataset or 'gsm8k_test' in dataset or 'math500' in dataset:
                label = math_equal(memoized_canonical_form(answer), memoized_canonical_form(ground_truth))
            else:
                label = (answer.strip().lower() == ground_truth.strip().lower())
            # Extract uncertainty scores
            unc = extract_unc(result.outputs[0])
            if unc is None:
                continue
                
            processed_data.append({
                'unique_id': sample['unique_id'],
                'label': label,
                'll': unc['ll'],
                'au': -unc['au'],
                'tu': -unc['tu'],
                'eu': -unc['eu'],
            })
            
        except Exception as e:
            continue

    if not processed_data:
        print(f"No samples processed successfully for seed {seed}")
        return None
        
    df = pd.DataFrame(processed_data)
    df.set_index('unique_id', inplace=True)
    
    # Calculate metrics
    metrics = {
        'seed': seed,
        'total_samples': len(df),
        'correct_samples': df['label'].sum(),
        'overall_accuracy': df['label'].mean()
    }
    
    # Available metrics
    eval_metrics = []
    for col in ['ll', 'au', 'tu', 'eu']:
        if col in df.columns and not df[col].isna().all():
            eval_metrics.append(col)
    
    # AUC Scores
    for metric in eval_metrics:
        auc_score = safe_auc(df['label'], df[metric])
        metrics[f'auc_{metric}'] = auc_score
    
    # AUPRC Scores
    for metric in eval_metrics:
        auprc_score = safe_auprc(df['label'], df[metric])
        metrics[f'auprc_{metric}'] = auprc_score
    
    # Top P% Accuracies
    p_values = [0.1, 0.25, 0.5, 0.75]
    for metric in eval_metrics:
        for p in p_values:
            try:
                acc = get_top_p_acc(df, p, metric)
                metrics[f'top{int(p*100)}_{metric}'] = acc
            except:
                metrics[f'top{int(p*100)}_{metric}'] = np.nan
    
    return metrics


def create_summary_report(df_metrics, results_dir, args, successful_seeds):
    """Create a comprehensive summary report"""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("MULTI-SEED EVALUATION SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {args.dataset}")
    report_lines.append(f"Model: {args.model}")
    report_lines.append(f"Results Subdirectory: {args.results_subdir}")
    report_lines.append(f"Successfully Evaluated Seeds: {successful_seeds}")
    report_lines.append(f"Total Seeds: {len(successful_seeds)}")
    report_lines.append("")
    
    # Overall accuracy
    acc_mean = df_metrics['overall_accuracy'].mean()
    acc_std = df_metrics['overall_accuracy'].std()
    report_lines.append("OVERALL ACCURACY:")
    report_lines.append(f"  Mean ± Std: {acc_mean:.4f} ± {acc_std:.4f}")
    report_lines.append(f"  Range: {df_metrics['overall_accuracy'].min():.4f} - {df_metrics['overall_accuracy'].max():.4f}")
    report_lines.append("")
    
    # AUC scores summary
    report_lines.append("AUC SCORES (Mean ± Std):")
    auc_cols = [col for col in df_metrics.columns if col.startswith('auc_')]
    for col in sorted(auc_cols):
        metric = col.replace('auc_', '').upper()
        values = df_metrics[col].dropna()
        if len(values) > 1:
            mean_val = values.mean() * 100
            std_val = values.std() * 100
            report_lines.append(f"  {metric:>8}: {mean_val:.2f}±{std_val:.2f}")
        elif len(values) == 1:
            report_lines.append(f"  {metric:>8}: {values.iloc[0]*100:.2f}")
    report_lines.append("")
    
    # AUPRC scores summary
    report_lines.append("AUPRC SCORES (Mean ± Std):")
    auprc_cols = [col for col in df_metrics.columns if col.startswith('auprc_')]
    for col in sorted(auprc_cols):
        metric = col.replace('auprc_', '').upper()
        values = df_metrics[col].dropna()
        if len(values) > 1:
            mean_val = values.mean() * 100
            std_val = values.std() * 100
            report_lines.append(f"  {metric:>8}: {mean_val:.2f}±{std_val:.2f}")
        elif len(values) == 1:
            report_lines.append(f"  {metric:>8}: {values.iloc[0]*100:.2f}")
    report_lines.append("")
    
    # Top 50% accuracy (highlighted)
    report_lines.append("TOP 50% ACCURACY (Mean ± Std):")
    top50_cols = [col for col in df_metrics.columns if col.startswith('top50_')]
    for col in sorted(top50_cols):
        metric = col.replace('top50_', '').upper()
        values = df_metrics[col].dropna()
        if len(values) > 1:
            mean_val = values.mean() * 100
            std_val = values.std() * 100
            report_lines.append(f"  {metric:>8}: {mean_val:.2f}±{std_val:.2f}")
        elif len(values) == 1:
            report_lines.append(f"  {metric:>8}: {values.iloc[0]*100:.2f}")
    report_lines.append("")
    
    # Best performing metrics
    report_lines.append("BEST PERFORMING METRICS:")
    
    if auc_cols:
        best_auc_col = max(auc_cols, key=lambda x: df_metrics[x].mean())
        best_auc_metric = best_auc_col.replace('auc_', '').upper()
        best_auc_value = df_metrics[best_auc_col].mean()
        report_lines.append(f"  Best AUC: {best_auc_metric} ({best_auc_value*100:.2f})")
    
    if top50_cols:
        best_top50_col = max(top50_cols, key=lambda x: df_metrics[x].mean())
        best_top50_metric = best_top50_col.replace('top50_', '').upper()
        best_top50_value = df_metrics[best_top50_col].mean() * 100
        report_lines.append(f"  Best Top 50%: {best_top50_metric} ({best_top50_value:.2f})")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_file = os.path.join(results_dir, 'evaluation_report.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print report
    print('\n'.join(report_lines))
    
    return report_file


def main():
    parser = argparse.ArgumentParser(description='multi-seed evaluation for uncertainty results')
    parser.add_argument('--dataset', default='math500', help='Dataset name')
    parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds to evaluate')
    parser.add_argument('--seeds_range', nargs=2, type=int, help='Range of seeds (start, end)')
    parser.add_argument('--model', default='qwen3b', help='Model name')
    parser.add_argument('--results_subdir', default='greedy_unc',
                       help='Results subdirectory')
    
    args = parser.parse_args()
    
    # Determine seeds to evaluate
    if args.seeds:
        seeds = args.seeds
    elif args.seeds_range:
        seeds = list(range(args.seeds_range[0], args.seeds_range[1] + 1))
    else:
        # Default seeds
        seeds = [96, 89, 64]
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = create_results_directory(args.model, args.dataset, timestamp)
    
    print(f"{'='*80}")
    print(f"MULTI-SEED UNCERTAINTY EVALUATION")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seeds: {seeds}")
    print(f"Results will be saved to: {results_dir}")
    
    # Evaluate all seeds
    all_metrics = []
    successful_seeds = []
    
    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")
        metrics = evaluate_single_seed(args.dataset, args.model, seed, args.results_subdir)
        if metrics:
            all_metrics.append(metrics)
            successful_seeds.append(seed)
            print(f"Seed {seed}: {metrics['correct_samples']}/{metrics['total_samples']} correct ({metrics['overall_accuracy']:.4f})")
        else:
            print(f"Seed {seed}: Failed to evaluate")
    
    if not all_metrics:
        print("\nNo seeds were successfully evaluated!")
        return
    
    # Convert to DataFrame for easier analysis
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save metadata
    metadata = save_metadata(results_dir, args, seeds, successful_seeds)
    print(f"\nMetadata saved to: {os.path.join(results_dir, 'metadata.json')}")
    
    # Save detailed results
    detailed_file = os.path.join(results_dir, 'detailed_results.csv')
    df_metrics.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Create and save summary statistics
    summary_data = []
    
    # Overall accuracy
    summary_data.append({
        'metric': 'overall_accuracy',
        'mean': df_metrics['overall_accuracy'].mean(),
        'std': df_metrics['overall_accuracy'].std(),
        'min': df_metrics['overall_accuracy'].min(),
        'max': df_metrics['overall_accuracy'].max(),
        'count': len(df_metrics)
    })
    
    # AUC scores
    auc_cols = [col for col in df_metrics.columns if col.startswith('auc_')]
    for col in sorted(auc_cols):
        values = df_metrics[col].dropna()
        if len(values) > 0:
            summary_data.append({
                'metric': col,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'count': len(values)
            })
    
    # AUPRC scores
    auprc_cols = [col for col in df_metrics.columns if col.startswith('auprc_')]
    for col in sorted(auprc_cols):
        values = df_metrics[col].dropna()
        if len(values) > 0:
            summary_data.append({
                'metric': col,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'count': len(values)
            })
    
    # Top p% accuracies
    for p in [10, 25, 50, 75]:
        top_cols = [col for col in df_metrics.columns if col.startswith(f'top{p}_')]
        for col in sorted(top_cols):
            values = df_metrics[col].dropna()
            if len(values) > 0:
                summary_data.append({
                    'metric': col,
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'count': len(values)
                })
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")

    # Create comprehensive report
    report_file = create_summary_report(df_metrics, results_dir, args, successful_seeds)
    print(f"Evaluation report saved to: {report_file}")
    
    print(f"\n{'='*80}")
    print(f"Multi-seed evaluation completed successfully!")
    print(f"All results saved in: {results_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
