#!/bin/bash

# =====================================================================
# Multi-GPU Scaling Test Script (Table 2)
# =====================================================================
# This script runs the scaling evaluation using multiple GPUs.
# It evaluates test-time scaling performance by subsampling candidate
# responses and measuring accuracy with different uncertainty-based
# selection strategies.
# =====================================================================

# Default parameters
MODEL="llama1b"
DATASET="math500"
SEED=96
P_VAL=0.1
N_SAMPLES="16,32,64,128,256,512"
N_PARTICLES=512
N_REPEATS=10
SCORE_TYPES="nau,neu,ntu,nll"
OUTPUT_DIR=""
NUM_GPUS=8
MAX_SAMPLES=""
RESULTS_BASE_DIR=""
MODEL_BASE_DIR="${MODEL_BASE_DIR:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --p_val)
            P_VAL="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --n_particles)
            N_PARTICLES="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --score_types)
            SCORE_TYPES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --results_base_dir)
            RESULTS_BASE_DIR="$2"
            shift 2
            ;;
        --model_base_dir)
            MODEL_BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL              Model name: llama1b, llama8b, qwen3b (default: llama1b)"
            echo "  --dataset DATASET          Dataset name (default: math500)"
            echo "  --seed SEED                Random seed (default: 96)"
            echo "  --p_val P_VAL              P value for top-p selection (default: 0.1)"
            echo "  --num_gpus NUM             Number of GPUs to use (default: 8)"
            echo "  --n_particles NUM          Number of particles used in generation (default: 512)"
            echo "  --max_samples NUM          Max samples for testing (default: all)"
            echo "  --score_types TYPES        Comma-separated score types (default: nau,neu,ntu,nll)"
            echo "  --output_dir DIR           Output directory (default: results/eval/scaling/)"
            echo "  --results_base_dir DIR     Base dir for pre-generated results"
            echo "  --model_base_dir DIR       Base dir for model checkpoints (for ce/dc scores)"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    echo "Error: CUDA is not available"
    exit 1
fi

# Check number of available GPUs
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "Using $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "==============================================="
echo "Multi-GPU Scaling Test Configuration"
echo "==============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "P value: $P_VAL"
echo "N samples: $N_SAMPLES"
echo "N particles: $N_PARTICLES"
echo "N repeats: $N_REPEATS"
echo "Score types: $SCORE_TYPES"
echo "Output directory: ${OUTPUT_DIR:-results/eval/scaling/ (default)}"
echo "Number of GPUs: $NUM_GPUS"
[ -n "$MAX_SAMPLES" ] && echo "Max samples: $MAX_SAMPLES"
[ -n "$RESULTS_BASE_DIR" ] && echo "Results base dir: $RESULTS_BASE_DIR"
[ -n "$MODEL_BASE_DIR" ] && echo "Model base dir: $MODEL_BASE_DIR"
echo "==============================================="

# Set environment variables for better multi-GPU performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1

# Build the command
CMD="python eval/eval_scaling_test_multi_gpu.py \
    --model $MODEL \
    --dataset $DATASET \
    --seed $SEED \
    --p_val $P_VAL \
    --n_samples_list $N_SAMPLES \
    --n_particles $N_PARTICLES \
    --n_repeats $N_REPEATS \
    --score_types $SCORE_TYPES \
    --num_gpus $NUM_GPUS"

[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output_dir $OUTPUT_DIR"
[ -n "$MAX_SAMPLES" ] && CMD="$CMD --max_samples $MAX_SAMPLES"
[ -n "$RESULTS_BASE_DIR" ] && CMD="$CMD --results_base_dir $RESULTS_BASE_DIR"
[ -n "$MODEL_BASE_DIR" ] && CMD="$CMD --model_base_dir $MODEL_BASE_DIR"

echo "Starting multi-GPU scaling test..."
echo "Running: $CMD"
echo ""

eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "==============================================="
    echo "Multi-GPU scaling test completed successfully!"
    echo "==============================================="
else
    echo "==============================================="
    echo "Multi-GPU scaling test failed with exit code: $EXIT_CODE"
    echo "==============================================="
    exit $EXIT_CODE
fi
