import os
import sys

# TensorFlow safety pre-allocations to prevent OOM conflicts with PyTorch (Repository Rule)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import time
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from run_rag_openbookqa import run_experiment
from openbookqa_rag_args import OpenBookQARAGArgs

# The 7 canonical benchmark architectures matching the full evaluation suite
MODELS = [
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'microsoft/Phi-3-medium-4k-instruct'
]


def main():
    start_total_time = time.time()
    print("=" * 85)
    print("OPENBOOKQA NEGATIVE CONTROL / RANDOM DISTRACTOR SWEEP (ALL 7 MODELS)")
    print("Evaluates RAG and layer-wise uncertainty dissipation under pure irrelevant noise (Hit@1 = 0%).")
    print(f"Total Target Models: {len(MODELS)}")
    for i, m in enumerate(MODELS, 1):
        print(f"  [{i}/7] {m}")
    print("=" * 85)

    completed_models = []
    failed_models = []

    for idx, model_name in enumerate(MODELS, 1):
        m_start = time.time()
        print(f"\n{'#' * 85}")
        print(f"# [{idx}/{len(MODELS)}] RUNNING NEGATIVE CONTROL DISTRACTOR RAG ON: {model_name}")
        print(f"{'#' * 85}")
        
        args = OpenBookQARAGArgs()
        args.llm_model = model_name
        args.retrieval_mode = 'distractor'
        args.top_k_retrieval = 1
        
        try:
            run_experiment(args)
            completed_models.append(model_name)
        except Exception as e:
            print(f"[ERROR] Failed Distractor run for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append((model_name, str(e)))
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        elapsed_m = (time.time() - m_start) / 60
        print(f"\n>>> [STATUS] Completed {model_name} in {elapsed_m:.2f} minutes.")

    total_elapsed_h = (time.time() - start_total_time) / 3600
    print("\n" + "=" * 85)
    print("NEGATIVE CONTROL DISTRACTOR RAG SWEEP EXECUTION SUMMARY")
    print(f"Total Duration     : {total_elapsed_h:.2f} hours")
    print(f"Successful Models ({len(completed_models)}/{len(MODELS)}):")
    for m in completed_models:
        print(f"  ✓ {m}")
    if failed_models:
        print(f"Failed Models ({len(failed_models)}/{len(MODELS)}):")
        for m, err in failed_models:
            print(f"  ✗ {m}: {err}")
    print("=" * 85)


if __name__ == '__main__':
    main()
