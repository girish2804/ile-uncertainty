#!/usr/bin/env python3
"""
Multi-GPU Scaling Test Script for Top-p Uncertainty Evaluation (Table 2)

This script distributes evaluation across multiple GPUs using data parallelism.
Each GPU gets a subset of the dataset and a copy of the model.
Results are aggregated across all GPUs for final statistics.

Usage:
    python eval/eval_scaling_test_multi_gpu.py \
        --model llama1b --dataset math500 --seed 96 --p_val 0.9 --num_gpus 4
"""

import argparse
import json
from glob import glob
import os
import sys
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset

# Project root setup (same pattern as eval_detect_multi_seed.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# NOTE: run.utils.math creates a multiprocessing.Manager() at import time,
# which conflicts with mp.set_start_method('spawn'). We import it lazily
# inside worker_process instead. qwen_math_parser is safe to import here.
from run.utils.qwen_math_parser import math_equal, extract_answer, strip_string

# Model name -> subdirectory mapping for model_base_dir
MODEL_PATH_MAP = {
    'llama1b': 'Llama-3.2-1B-Instruct',
    'llama8b': 'Meta-Llama-3.1-8B-Instruct',
    'qwen3b': 'Qwen2.5-3B-Instruct',
}


def extract(string):
    """Extract answer from model output string"""
    try:
        return strip_string(extract_answer(string, 'math'))
    except Exception:
        return string


def softmax(x):
    """Compute softmax with numerical stability"""
    x = np.array(x)
    x = np.exp(x - np.max(x))
    return x / np.sum(x, axis=-1, keepdims=True)


class LazyDataLoader:
    """Lazy loading of uncertainty data files from pre-generated results"""

    def __init__(self, model: str, dataset: str, seed: int,
                 p: int = 512, results_base_dir: str = None):
        self.model = model
        self.dataset = dataset
        self.seed = seed
        self.p = p

        if results_base_dir is None:
            results_base_dir = os.path.join(BASE_DIR, 'results', 'inference_scaling')

        self.base_path = os.path.join(
            results_base_dir,
            f'{model}_results_vllm_pg', dataset,
            f'seed{seed}', f'p{p}', 'single_greedy_unc'
        )

        self.files = glob(os.path.join(self.base_path, '*.json'))
        self.file_map = {}
        self.cache = {}

        for file_path in self.files:
            filename = os.path.basename(file_path)
            if not filename.startswith('batch'):
                self.file_map[filename] = file_path

        print(f"Found {len(self.file_map)} files available for lazy loading")

        if len(self.file_map) == 0:
            print(f"Warning: No files found in {self.base_path}")

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]

        if key not in self.file_map:
            raise KeyError(f"File {key} not found")

        with open(self.file_map[key], 'r') as f:
            data = json.load(f)
            self.cache[key] = data
            return data

    def keys(self):
        return self.file_map.keys()

    def clear_cache(self):
        """Clear cache to free memory"""
        self.cache.clear()


def get_unc_scores(response):
    """Extract uncertainty scores from a response object"""
    unc_score = dict(response['uncertainties'])
    unc_score['nll'] = response['logprobs']['nll']
    unc_score['ll'] = response['logprobs']['ll']
    return unc_score


def calculate_self_certainty(logits, epsilon=1e-8):
    """Calculate Self-certainty (negative entropy) - Memory optimized version"""
    batch_size, seq_len, vocab_size = logits.shape

    probs = F.softmax(logits, dim=-1)
    probs.add_(epsilon)

    log_V = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))

    probs.log_()
    probs.add_(log_V)

    entropy_per_position = probs.sum(dim=-1)
    return -entropy_per_position / vocab_size


def calculate_token_confidence(logits, k=20, epsilon=1e-8):
    """Calculate Token Confidence (C_i)"""
    batch_size, seq_len, vocab_size = logits.shape

    k = min(k, vocab_size)
    probs = F.softmax(logits, dim=-1)

    topk_probs, _ = torch.topk(probs, k=k, dim=-1, largest=True)
    topk_probs = torch.clamp(topk_probs, min=epsilon)

    log_topk_probs = torch.log(topk_probs)
    token_confidence = -log_topk_probs.mean(dim=-1)

    return token_confidence


def _get_sample_id(sample):
    """Get a consistent unique ID string from a dataset sample"""
    if 'unique_id' in sample:
        raw_id = sample['unique_id']
    elif 'metadata' in sample and 'source_index' in sample['metadata']:
        raw_id = str(sample['metadata']['source_index']) + '.json'
    else:
        raise KeyError("Sample missing 'unique_id' or 'metadata.source_index'")
    return str(raw_id).replace('/', '_')


def run_experiments_for_sample(sample: Dict, data_loader: LazyDataLoader,
                               model_entropy, tokenizer,
                               p_value: float,
                               n_samples_list: List[int], n_repeats: int,
                               base_seed: int,
                               score_types: List[str] = None,
                               mode: str = 'mean',
                               canonical_form_func=None,
                               math_equal_func=None) -> Dict[int, List[Dict[str, float]]]:
    """Run all experiments for a single sample across different n_samples values.

    For each n_samples value, subsamples from the pool of pre-generated responses,
    ranks them by each score type, selects top-p, and computes majority vote (maj)
    and weighted best-of-n (wbon) accuracy.
    """
    if score_types is None:
        score_types = ['nau', 'neu', 'ntu', 'nll']
    if math_equal_func is None:
        math_equal_func = math_equal

    unique_id = _get_sample_id(sample)

    if unique_id not in data_loader.keys():
        return {}

    try:
        responses_data = data_loader[unique_id]['outputs']
    except Exception:
        return {}

    if not responses_data:
        return {}

    # Pre-extract all answers and scores once
    all_answers = []
    all_scores = {st: [] for st in score_types}
    needs_entropy_model = ('ce' in score_types or 'dc' in score_types)

    for response in responses_data:
        scores_dict = get_unc_scores(response)
        if scores_dict is None:
            continue

        # Compute ce/dc scores on-the-fly if requested
        if needs_entropy_model and model_entropy is not None:
            inputs = tokenizer(response['text'], return_tensors="pt").to(model_entropy.device)
            with torch.no_grad():
                outputs = model_entropy(**inputs)
                logits = outputs.logits
            scores_dict['ce'] = torch.mean(calculate_self_certainty(logits)).item()
            scores_dict['dc'] = torch.mean(calculate_token_confidence(logits, k=40)).item()

        answer = extract(response['text'])
        all_answers.append(answer)

        for score_type in score_types:
            if score_type in ['nll', 'ce', 'dc', 'll']:
                all_scores[score_type].append(scores_dict[score_type])
            else:
                # Negate uncertainty scores so higher = better (lower uncertainty)
                all_scores[score_type].append(-scores_dict[score_type])

    if len(all_answers) == 0:
        return {}

    gt_answer = canonical_form_func(sample['answer'])

    # Run experiments for all n_samples values
    max_n = max(n_samples_list)
    sample_results = {}

    for n_samples in n_samples_list:
        current_repeats = 1 if n_samples == max_n else n_repeats
        experiment_results = []

        for experiment_id in range(current_repeats):
            experiment_seed = hash((base_seed, unique_id, n_samples, experiment_id)) % (2**32)

            if n_samples < len(all_answers):
                np.random.seed(experiment_seed)
                indices = np.random.choice(len(all_answers), size=n_samples, replace=False)
                sampled_answers = [all_answers[i] for i in indices]
                sampled_scores = {
                    st: [all_scores[st][i] for i in indices] for st in score_types
                }
            else:
                sampled_answers = all_answers
                sampled_scores = all_scores

            experiment_result = {}

            for score_type in score_types:
                if len(sampled_scores[score_type]) == 0:
                    continue

                # Sort by scores and select top-p fraction
                sorted_indices = np.argsort(sampled_scores[score_type])[::-1]
                top_k = max(1, int(len(sampled_answers) * p_value))
                top_indices = sorted_indices[:top_k]

                top_answers = [sampled_answers[i] for i in top_indices]
                top_scores = [sampled_scores[score_type][i] for i in top_indices]

                # Majority voting (maj)
                answer_counts = defaultdict(int)
                for ans in top_answers:
                    canonical = canonical_form_func(ans)
                    answer_counts[canonical] += 1
                maj_answer = max(answer_counts, key=answer_counts.get) if answer_counts else ""

                # Weighted best-of-n (wbon)
                normalized_scores = softmax(np.array(top_scores))
                weighted_counts = defaultdict(float)
                for ans, weight in zip(top_answers, normalized_scores):
                    canonical = canonical_form_func(ans)
                    weighted_counts[canonical] += weight
                wbon_answer = max(weighted_counts, key=weighted_counts.get) if weighted_counts else ""

                experiment_result[f'{score_type}_maj'] = math_equal_func(gt_answer, maj_answer)
                experiment_result[f'{score_type}_wbon'] = math_equal_func(gt_answer, wbon_answer)

            experiment_results.append(experiment_result)

        sample_results[n_samples] = experiment_results

    return sample_results


def worker_process(gpu_id: int, dataset_subset: List, shared_results: Dict,
                   model: str, dataset: str, seed: int, p_value: float,
                   n_samples_list: List[int], n_repeats: int,
                   score_types: List[str], mode: str,
                   results_base_dir: str = None, model_base_dir: str = None,
                   n_particles: int = 512):
    """Worker function for each GPU process"""

    # Lazy import: run.utils.math creates a multiprocessing.Manager() at module
    # level which conflicts with the spawn start method used by this script.
    from run.utils.math import memoized_canonical_form

    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    print(f"GPU {gpu_id}: Processing {len(dataset_subset)} samples")

    # Initialize data loader
    data_loader = LazyDataLoader(
        model=model, dataset=dataset, seed=seed,
        p=n_particles, results_base_dir=results_base_dir
    )

    # Only load model/tokenizer if ce or dc scores are requested
    needs_entropy_model = ('ce' in score_types or 'dc' in score_types)

    if needs_entropy_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_base_dir is None:
            model_base_dir = os.environ.get('MODEL_BASE_DIR')
            if model_base_dir is None:
                raise ValueError("MODEL_BASE_DIR environment variable must be set when using 'ce' or 'dc' score types")

        if model not in MODEL_PATH_MAP:
            raise ValueError(f"Model '{model}' not supported. Choose from: {list(MODEL_PATH_MAP.keys())}")

        model_path = os.path.join(model_base_dir, MODEL_PATH_MAP[model])
        print(f"GPU {gpu_id}: Loading entropy model from {model_path}")
        model_entropy = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model_entropy = None
        tokenizer = None

    # Initialize worker results
    worker_results = {
        'successful_samples': 0,
        'failed_samples': [],
        'sample_experiments': {n_samples: {} for n_samples in n_samples_list}
    }

    for i, sample in enumerate(tqdm(dataset_subset, desc=f"GPU {gpu_id}")):
        sample_results = run_experiments_for_sample(
            sample=sample,
            data_loader=data_loader,
            model_entropy=model_entropy,
            tokenizer=tokenizer,
            p_value=p_value,
            n_samples_list=n_samples_list,
            n_repeats=n_repeats,
            base_seed=seed,
            score_types=score_types,
            mode=mode,
            canonical_form_func=memoized_canonical_form,
            math_equal_func=math_equal,
        )

        unique_id = _get_sample_id(sample)

        if sample_results:
            for n_samples, experiments in sample_results.items():
                worker_results['sample_experiments'][n_samples][unique_id] = experiments
            worker_results['successful_samples'] += 1
        else:
            worker_results['failed_samples'].append(unique_id)

        if i % 50 == 0:
            data_loader.clear_cache()

    shared_results[gpu_id] = worker_results
    print(f"GPU {gpu_id}: Completed processing {worker_results['successful_samples']} samples")


def split_dataset(dataset, num_gpus: int) -> List[List]:
    """Split dataset into roughly equal chunks for each GPU"""
    dataset_size = len(dataset)
    chunk_size = dataset_size // num_gpus
    remainder = dataset_size % num_gpus

    chunks = []
    start_idx = 0

    for i in range(num_gpus):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunk_dataset = dataset.select(range(start_idx, end_idx))
        chunks.append([sample for sample in chunk_dataset])
        start_idx = end_idx

    return chunks


def aggregate_results(worker_results: Dict, n_samples_list: List[int],
                      score_types: List[str], model: str, dataset: str,
                      seed: int, p_value: float, mode: str) -> Dict:
    """Aggregate results from all worker processes"""

    scaling_results = {
        'metadata': {
            'model': model,
            'dataset': dataset,
            'seed': seed,
            'p_value': p_value,
            'n_samples_list': n_samples_list,
            'score_types': score_types,
            'mode': mode,
            'num_gpus': len(worker_results)
        },
        'results': {}
    }

    for n_samples in n_samples_list:
        scaling_results['results'][n_samples] = {
            'statistics': {},
            'sample_experiments': {},
            'n_experiments': 0
        }

    total_successful = 0
    total_failed = []

    for gpu_id, worker_result in worker_results.items():
        total_successful += worker_result['successful_samples']
        total_failed.extend(worker_result['failed_samples'])

        for n_samples in n_samples_list:
            scaling_results['results'][n_samples]['sample_experiments'].update(
                worker_result['sample_experiments'][n_samples]
            )

    print(f"Total successful samples: {total_successful}")
    print(f"Total failed samples: {len(total_failed)}")

    for n_samples in n_samples_list:
        print(f"\n--- Calculating statistics for n_samples = {n_samples} ---")

        sample_experiments = scaling_results['results'][n_samples]['sample_experiments']

        if not sample_experiments:
            print(f"  No successful experiments for n_samples = {n_samples}")
            continue

        stats = {}
        for score_type in score_types:
            for method in ['maj', 'wbon']:
                key = f'{score_type}_{method}'

                all_individual_results = []

                for _, sample_exps in sample_experiments.items():
                    sample_accuracies = []
                    for result in sample_exps:
                        if key in result:
                            sample_accuracies.append(result[key])

                    if sample_accuracies:
                        all_individual_results.append(sample_accuracies)

                if all_individual_results:
                    all_individual_results = np.array(all_individual_results, dtype=int)
                    acc = np.mean(all_individual_results, axis=0)
                    mean_acc = np.mean(acc)
                    std_acc = np.std(acc)

                    stats[key] = {
                        'mean': mean_acc * 100,
                        'std': std_acc * 100,
                        'count': len(all_individual_results)
                    }

        scaling_results['results'][n_samples]['statistics'] = stats
        scaling_results['results'][n_samples]['n_experiments'] = len(sample_experiments)

        total_experiments = sum(len(exp) for exp in sample_experiments.values())
        print(f"  Results summary for n_samples = {n_samples} "
              f"({len(sample_experiments)} samples, {total_experiments} total experiments):")
        for key, stat in stats.items():
            print(f"    {key}: {stat['mean']:.2f}% +/- {stat['std']:.2f}% (n={stat['count']})")

    return scaling_results


def run_scaling_test_multi_gpu(model: str, dataset: str, seed: int, p_value: float,
                               n_samples_list: List[int] = None,
                               n_repeats: int = 10,
                               score_types: List[str] = None,
                               output_dir: str = None,
                               mode: str = 'mean', max_samples: int = None,
                               num_gpus: int = 4,
                               results_base_dir: str = None,
                               model_base_dir: str = None,
                               n_particles: int = 512) -> Dict:
    """
    Run scaling test with multiple GPUs.

    Args:
        model: Model name (llama1b, llama8b, qwen3b)
        dataset: Dataset name (math500, gsm8k_test, deepscaler, etc.)
        seed: Base random seed
        p_value: P value for top-p selection
        n_samples_list: List of n_samples values to test
        n_repeats: Number of repetitions for each n_samples (except max)
        score_types: List of uncertainty score types
        output_dir: Output directory for results
        mode: Uncertainty aggregation mode
        max_samples: Maximum number of samples to evaluate (None for all)
        num_gpus: Number of GPUs to use
        results_base_dir: Base directory for pre-generated results
        model_base_dir: Base directory for model checkpoints
        n_particles: Number of particles used during generation
    """
    if n_samples_list is None:
        n_samples_list = [16, 32, 64, 128, 256, 512]
    if score_types is None:
        score_types = ['nau', 'neu', 'ntu', 'nll']
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'results', 'eval', 'scaling')

    print("=" * 70)
    print("Multi-GPU Top-p Uncertainty Scaling Test")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")
    print(f"P value: {p_value}")
    print(f"N samples: {n_samples_list}")
    print(f"N repeats: {n_repeats}")
    print(f"N particles: {n_particles}")
    print(f"Score types: {score_types}")
    print(f"Mode: {mode}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Results base dir: {results_base_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available. "
              f"Using {available_gpus}.")
        num_gpus = available_gpus

    # Load ground truth dataset
    dataset_path = os.path.join(BASE_DIR, 'datasets', f'{dataset}.jsonl')
    gt = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(gt)} samples from {dataset}")

    if max_samples is not None:
        gt = gt.select(range(min(max_samples, len(gt))))
        print(f"Limited to first {len(gt)} samples for testing")

    dataset_chunks = split_dataset(gt, num_gpus)
    print(f"Split dataset into {num_gpus} chunks: {[len(chunk) for chunk in dataset_chunks]}")

    os.makedirs(output_dir, exist_ok=True)

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    shared_results = manager.dict()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, dataset_chunks[gpu_id], shared_results,
                  model, dataset, seed, p_value, n_samples_list, n_repeats,
                  score_types, mode, results_base_dir, model_base_dir, n_particles)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n=== All GPU processes completed ===")

    # Aggregate results
    scaling_results = aggregate_results(
        dict(shared_results), n_samples_list, score_types,
        model, dataset, seed, p_value, mode
    )

    # Save summary CSV
    summary_data = []
    for n_samples, results in scaling_results['results'].items():
        for score_method, stats in results['statistics'].items():
            summary_data.append({
                'model': model,
                'dataset': dataset,
                'seed': seed,
                'p_value': p_value,
                'mode': mode,
                'n_particles': n_particles,
                'n_samples': n_samples,
                'score_method': score_method,
                'mean_accuracy': stats['mean'],
                'std_accuracy': stats['std'],
                'n_repeats': stats['count'],
                'num_gpus': num_gpus
            })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(
        output_dir,
        f"{model}_{dataset}_seed{seed}_p{p_value:.1f}_{mode}_scaling_summary.csv"
    )
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV saved to: {summary_csv}")

    # Print final scaling summary table
    print(f"\n{'=' * 80}")
    print(f"Final Scaling Summary  |  Model: {model}  Dataset: {dataset}  "
          f"P: {p_value}  Mode: {mode}  GPUs: {num_gpus}")
    print("-" * 80)
    print(f"{'N_Samples':<10} {'Score_Method':<15} {'Mean_Acc(%)':<12} "
          f"{'Std_Acc(%)':<11} {'N_Exp':<6}")
    print("-" * 80)

    for n_samples in sorted(scaling_results['results'].keys()):
        results = scaling_results['results'][n_samples]
        for score_method, stats in sorted(results['statistics'].items()):
            print(f"{n_samples:<10} {score_method:<15} {stats['mean']:<12.2f} "
                  f"{stats['std']:<11.2f} {stats['count']:<6}")

    print("=" * 80)

    return scaling_results


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-GPU scaling test for top-p uncertainty sampling (Table 2)'
    )
    parser.add_argument('--model', type=str, default='llama1b',
                        choices=['llama1b', 'llama8b', 'qwen3b'],
                        help='Model name (default: llama1b)')
    parser.add_argument('--dataset', type=str, default='math500',
                        help='Dataset name (default: math500)')
    parser.add_argument('--seed', type=int, default=96,
                        help='Random seed (default: 96)')
    parser.add_argument('--p_val', type=float, default=0.9,
                        help='P value for top-p selection (default: 0.9)')
    parser.add_argument('--n_samples_list', type=str, default='16,32,64,128,256,512',
                        help='Comma-separated n_samples values (default: 16,32,64,128,256,512)')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of repetitions per n_samples except max (default: 10)')
    parser.add_argument('--n_particles', type=int, default=512,
                        help='Number of particles used during generation (default: 512)')
    parser.add_argument('--score_types', type=str, default='nau,neu,ntu,nll',
                        help='Comma-separated score types (default: nau,neu,ntu,nll)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/eval/scaling/)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (default: None for all)')
    parser.add_argument('--mode', type=str, default='mean',
                        choices=['mean', 'first10p', 'last10p', 'llmax10p', 'llmin10p'],
                        help='Uncertainty aggregation mode (default: mean)')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='Number of GPUs to use (default: 4)')
    parser.add_argument('--results_base_dir', type=str, default=None,
                        help='Base directory for pre-generated results '
                             '(default: results/inference_scaling/)')
    parser.add_argument('--model_base_dir', type=str, default=None,
                        help='Base directory for model checkpoints, used only for '
                             'ce/dc score types (default: MODEL_BASE_DIR env var)')

    args = parser.parse_args()

    n_samples_list = [int(x.strip()) for x in args.n_samples_list.split(',')]
    score_types = [x.strip() for x in args.score_types.split(',')]

    run_scaling_test_multi_gpu(
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        p_value=args.p_val,
        n_samples_list=n_samples_list,
        n_repeats=args.n_repeats,
        score_types=score_types,
        output_dir=args.output_dir,
        mode=args.mode,
        max_samples=args.max_samples,
        num_gpus=args.num_gpus,
        results_base_dir=args.results_base_dir,
        model_base_dir=args.model_base_dir,
        n_particles=args.n_particles,
    )

    print("\nMulti-GPU scaling test completed successfully!")


if __name__ == "__main__":
    main()
