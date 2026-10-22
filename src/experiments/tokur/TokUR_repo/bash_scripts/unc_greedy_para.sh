#!/bin/bash

# =====================================================================
# Uncertainty-based Greedy Response Generation Script
# =====================================================================
# This script runs uncertainty-based greedy response generation across
# multiple GPUs, particles, and seeds for parallel processing.
# =====================================================================

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# =====================================================================
# Configuration Parameters
# =====================================================================

# GPU Configuration
readonly NUM_GPUS=8
readonly GPU_IDS=(0 1 2 3 4 5 6 7)  # Adjust based on available GPUs

# Experiment Parameters
readonly NUM_PARTICLES=(512)
readonly SEEDS=(96)

# Dataset Configuration
readonly DATASET="math500"  # Options: math500, gsm8k_test, deepscaler, orz_balanced_5500, acre, zebra_puzzles
readonly DATASET_START=0
readonly DATASET_END=500

# Model Configuration
readonly MODEL="llama1b"  # Options: llama1b, llama8b, qwen3b

# Set MODEL_BASE_DIR to the directory containing your model checkpoints.
# You can download TFB models from HuggingFace:
#   - n1h111sm/TFB-Llama3.2-1B-Instruct
#   - n1h111sm/TFB-Meta-Llama-3.1-8B-Instruct
#   - n1h111sm/TFB-Qwen2.5-3B-Instruct
readonly MODEL_BASE_DIR="${MODEL_BASE_DIR:-./models}"

# Set model path based on MODEL variable
if [[ "$MODEL" == "llama1b" ]]; then
    readonly MODEL_PATH="${MODEL_BASE_DIR}/TFB-Llama3.2-1B-Instruct"
elif [[ "$MODEL" == "llama8b" ]]; then
    readonly MODEL_PATH="${MODEL_BASE_DIR}/TFB-Meta-Llama-3.1-8B-Instruct"
elif [[ "$MODEL" == "qwen3b" ]]; then
    readonly MODEL_PATH="${MODEL_BASE_DIR}/TFB-Qwen2.5-3B-Instruct"
else
    echo "Error: Invalid model specified: $MODEL" >&2
    echo "Available models: llama1b, llama8b, qwen3b" >&2
    exit 1
fi

readonly OUTPUT_BASE_DIR="./results/inference_scaling/${MODEL}_results_vllm_pg"

# Script Configuration
readonly DATASET_PATH="datasets/${DATASET}.jsonl"

# =====================================================================
print_configuration() {
    echo "📋 Experiment Configuration:"
    echo "  Dataset: $DATASET"
    echo "  Dataset Range: $DATASET_START - $DATASET_END"
    echo "  Number of GPUs: $NUM_GPUS"
    echo "  GPU IDs: ${GPU_IDS[*]}"
    echo "  Particles: ${NUM_PARTICLES[*]}"
    echo "  Seeds: ${SEEDS[*]}"
    echo "  Model: $MODEL"
    echo "  Model Path: $MODEL_PATH"
    echo "  Output Directory: $OUTPUT_BASE_DIR"
    echo ""
}

# =====================================================================
# Main Execution
# =====================================================================

main() {
    echo "🚀 Starting Uncertainty-based Greedy Response Generation"
    echo "======================================================"
    
    print_configuration
    
    # Calculate dataset chunking
    local total=$((DATASET_END - DATASET_START))
    local chunk_size=$((total / NUM_GPUS))
    
    echo "📊 Processing $total samples with chunk size: $chunk_size"
    echo ""
    
    local job_count=0
    local total_jobs=$((${#NUM_PARTICLES[@]} * ${#SEEDS[@]} * NUM_GPUS))
    
    for particles in "${NUM_PARTICLES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "🔧 Starting experiments with $particles particles, seed $seed"
            
            for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
                local gpu=${GPU_IDS[$gpu_idx]}
                local chunk_start=$((DATASET_START + gpu_idx * chunk_size))
                local chunk_end
                
                # Handle the last GPU to process remaining samples
                if [[ $gpu_idx -eq $((NUM_GPUS - 1)) ]]; then
                    chunk_end=$DATASET_END
                else
                    chunk_end=$((chunk_start + chunk_size))
                fi
                
                local output_dir="${OUTPUT_BASE_DIR}/${DATASET}/seed${seed}/p${particles}/single_greedy_unc"
                
                # Create output directory
                mkdir -p "$output_dir"
                
                job_count=$((job_count + 1))
                echo "🎯 Job $job_count/$total_jobs - GPU $gpu: samples $chunk_start-$chunk_end"
                
                # Launch the job
                CUDA_VISIBLE_DEVICES=$gpu \
                python run/greedy_responses_unc.py \
                    --dataset-path "$DATASET_PATH" \
                    --dataset-start "$chunk_start" \
                    --dataset-end "$chunk_end" \
                    --model-path "$MODEL_PATH" \
                    --output-dir "$output_dir" \
                    --n-particles "$particles" \
                    --seed "$seed" \
                    > "${output_dir}/gpu${gpu}_${chunk_start}_${chunk_end}.log" 2>&1 &
                
                echo "   📝 Log file: ${output_dir}/gpu${gpu}_${chunk_start}_${chunk_end}.log"
            done
            
            echo "⏳ Waiting for all GPU jobs to complete..."
            wait
            echo "✅ All jobs completed for $particles particles, seed $seed"
            echo ""
        done
    done
    
    echo "🎉 All experiments completed successfully!"
    echo "📁 Results saved to: $OUTPUT_BASE_DIR"
}

# =====================================================================
# Script Entry Point
# =====================================================================

# Only run main function if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi