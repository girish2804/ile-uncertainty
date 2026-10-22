#!/bin/bash

# multi-seed evaluation script with organized result management
# Usage: ./eval_detect.sh [dataset] [model] [seeds...]

# Default parameters
DATASET=${1:-"leg-counting"}
MODEL=${2:-"qwen3b"}
RESULTS_SUBDIR="greedy_unc"

# Default seeds if not provided
if [ $# -le 2 ]; then
    SEEDS=(96 89 64)
else
    # Use provided seeds (skip first two arguments)
    SEEDS=("${@:3}")
fi

echo "=========================================="
echo "Multi-Seed Uncertainty Evaluation"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Seeds: ${SEEDS[*]}"
echo "Results subdir: $RESULTS_SUBDIR"
echo ""

# Run Python evaluation script
echo ""
echo "🚀 Running multi-seed evaluation..."
python eval/eval_detect_multi_seed.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --results_subdir "$RESULTS_SUBDIR" \
    --seeds "${SEEDS[@]}"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "📁 Results are organized in:"
    echo "   eval/results/{model}_{dataset}_{timestamp}/"
    echo ""
    echo "📄 Files generated:"
    echo "   - detailed_results.csv      (raw data)"
    echo "   - summary_statistics.csv    (mean±std)"
    echo "   - evaluation_report.txt     (human-readable)"
    echo "   - metadata.json            (run information)"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
    echo "Check the error messages above."
    echo "=========================================="
fi

exit $EXIT_CODE
