import os
import sys

# TensorFlow safety pre-allocations to prevent OOM conflicts with PyTorch (Repository Rule)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import copy
import gc
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from run_rag_openbookqa import run_experiment
from openbookqa_rag_args import OpenBookQARAGArgs


def main():
    # Read the active configuration directly from openbookqa_rag_args.py
    # Change the model in openbookqa_rag_args.py (self.llm_model = [...][idx])
    args = OpenBookQARAGArgs()
    target_model = args.llm_model
    
    print("=" * 85)
    print("OPENBOOKQA PAIRED ABLATION RUNNER (MODEL-BY-MODEL CO-EXECUTION)")
    print(f"Target Model (Active in openbookqa_rag_args.py): {target_model}")
    print("Execution Plan:")
    print("  • Condition A: BM25-Optimized (Top-3 Facts, Query Distillation)")
    print("  • Condition B: Dense Top-1 (Cleanest Semantic Fact, Zero Distractors)")
    print("=" * 85)
    
    # 1. Run BM25-Optimized Ablation
    print(f"\n{'#' * 85}")
    print(f"# [1/2] RUNNING BM25-OPTIMIZED ON: {target_model}")
    print(f"{'#' * 85}")
    bm25_args = copy.deepcopy(args)
    bm25_args.retrieval_mode = 'bm25_opt'
    bm25_args.top_k_retrieval = 3
    run_experiment(bm25_args)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 2. Run Dense Top-1 Ablation
    print(f"\n{'#' * 85}")
    print(f"# [2/2] RUNNING DENSE TOP-1 ON: {target_model}")
    print(f"{'#' * 85}")
    top1_args = copy.deepcopy(args)
    top1_args.retrieval_mode = 'dense'
    top1_args.top_k_retrieval = 1
    run_experiment(top1_args)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\n" + "=" * 85)
    print(f"ALL PAIRED ABLATIONS FOR {target_model} COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == '__main__':
    main()
