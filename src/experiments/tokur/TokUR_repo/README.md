# TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2505.11737-b31b1b.svg)](https://arxiv.org/abs/2505.11737)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is the official implementation of the ICLR 2026 paper:

> **TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning**
>
> Tunyu Zhang\*, Haizhou Shi\*, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris Metaxas, Hao Wang
>
> \*Equal contribution

## Overview

TokUR is a **training-free** token-level uncertainty estimation framework for LLM reasoning. It introduces low-rank random weight perturbation during LLM decoding to generate predictive distributions for token-level uncertainty estimation, and aggregates these quantities to capture the semantic uncertainty of generated responses.

**Key features:**
- **Token-level uncertainty decomposition** into aleatoric uncertainty (data randomness) and epistemic uncertainty (model uncertainty) with theoretical guarantees.
- **Training-free**: Works with any pre-trained LLM without retraining or fine-tuning.
- **Practical applications**: Incorrect reasoning path detection, high-quality solution selection, and uncertainty-guided generation for test-time scaling.
- **vLLM integration**: Efficient batched inference via seamless vLLM support.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Wang-ML-Lab/TokUR.git
cd TokUR
```

### 2. Install the forked vLLM (required)

TokUR requires a [forked version of vLLM (v0.7.3)](https://github.com/haizhou-shi/vllm) that supports Bayesian weight perturbation during decoding:

```bash
git clone https://github.com/haizhou-shi/vllm.git
cd vllm
export VLLM_COMMIT=61c6a5a79664882a8ab1c9af3ff78677911516dc # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Bayesian Transformer package

```bash
cd bayesian_transformer
pip install -e .
cd ..
```



> **Important:** The `enforce_eager=True` flag must be used when initializing the vLLM model to disable CUDAGraph compilation, which would otherwise disable the Bayesian sampling process.

## Quick Start

```python
import bayesian_transformer
from vllm import LLM, SamplingParams

# Load a TFB (Training-Free Bayesian) model
# Use a HuggingFace model ID or a local path to a converted model
model = LLM(
    "/path/to/TFB-Llama-3.2-1B-Instruct",  # or "n1h111sm/TFB-Qwen2.5-3B-Instruct"
    enforce_eager=True,
)

# Generate with uncertainty estimation
prompt = "Solve: What is 15% of 240?"
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    logprobs=1,
)
output = model.generate(prompt, sampling_params)
print(output[0].outputs[0].text)
```

## Supported Models

| Model | HuggingFace ID | Architecture |
|-------|---------------|--------------|
| TFB-Qwen2.5-3B-Instruct | `n1h111sm/TFB-Qwen2.5-3B-Instruct` | Qwen 2.5 |

> **Note:** Due to licensing restrictions, we cannot publicly release the TFB weights for Llama models. You can convert Llama (or any other supported model) to a TFB model locally using the conversion script below.

## Converting Base Models to TFB

We provide `convert_to_tfb.py` to convert any supported base model into a TFB model by computing SVD basis vectors for the attention layers (`q_proj`, `v_proj`).

```bash
# Convert a Llama model
python convert_to_tfb.py \
    --model-path /path/to/Meta-Llama-3.2-1B-Instruct \
    --output-path /path/to/TFB-Llama-3.2-1B-Instruct \
    --architecture llama \
    --rank 8

# Convert a Qwen2 model
python convert_to_tfb.py \
    --model-path /path/to/Qwen2.5-3B-Instruct \
    --output-path /path/to/TFB-Qwen2.5-3B-Instruct \
    --architecture qwen2 \
    --rank 8
```

**Arguments:**
- `--model-path`: Path to the base HuggingFace model directory (must contain `.safetensors` files)
- `--output-path`: Where to save the converted TFB model
- `--architecture`: Model architecture (`llama` or `qwen2`)
- `--rank`: Rank for low-rank basis vectors (default: 8)
- `--bayes-noise`: Noise direction, `right` (default) or `left`

The script handles both single-file and sharded (multi-file) models automatically. The converted model can be loaded directly with vLLM or HuggingFace as shown in the Quick Start section.

## Data Preparation

We provide the datasets used in our experiments via HuggingFace. To download them:

```bash
cd datasets
python download_data.py
cd ..
```

This downloads **MATH500**, **GSM8K** (test set), **DeepScaleR** (subset), and **Leg Counting** (subset) in JSONL format.

## Reproducing Results

### Incorrect Reasoning Path Detection (Table 1-3)

**Step 1: Generate responses with uncertainty estimation**

Run greedy decoding with TokUR uncertainty on multiple GPUs:

```bash
# Set MODEL_BASE_DIR to your local model directory
export MODEL_BASE_DIR=/path/to/your/models

bash bash_scripts/unc_greedy_single_para_batch.sh
```

Or run manually:

```bash
CUDA_VISIBLE_DEVICES=$GPU python run/greedy_unc_single_batch_refine.py \
    --dataset-path "datasets/math500.jsonl" \
    --dataset-start 0 \
    --dataset-end 500 \
    --model-path /path/to/TFB-Llama3.2-1B-Instruct \
    --output-dir ./results/llama1b_results_vllm_pg/math500/seed96/greedy_unc \
    --seed 96 \
    --batch-size 16
```

**Step 2: Evaluate uncertainty quality**

```bash
bash bash_scripts/eval_detect.sh math500 llama1b 96 89 64
```

Or run the evaluation script directly:

```bash
python eval/eval_detect_multi_seed.py \
    --dataset "math500" \
    --model "llama1b" \
    --results_subdir "greedy_unc" \
    --seeds 96 89 64
```

Results (AUROC, AUPRC, Top-50% ACC) are saved to `results/eval/`.

### Test-Time Scaling (Table 4)

**Step 1: Generate multiple candidate responses**

```bash
export MODEL_BASE_DIR=/path/to/your/models

bash bash_scripts/unc_greedy_para.sh
```

**Step 2: Evaluate scaling performance**

```bash
bash bash_scripts/run_scaling_multi_gpu.sh \
    --model llama1b \
    --dataset math500 \
    --seed 96
```

## Project Structure

```
TokUR/
├── bayesian_transformer/          # Installable package for TFB models
│   ├── bayesian_transformer/
│   │   ├── __init__.py            # Auto-registration of models
│   │   ├── config.py              # BayesianLM configuration
│   │   ├── layers.py              # Low-rank Bayesian linear layers (core)
│   │   ├── model.py               # Bayesian wrapper for HuggingFace models
│   │   └── vllm_models/           # vLLM-optimized implementations
│   │       ├── tfb_llama.py       # TFB Llama (transformers)
│   │       ├── tfb_llama_vllm.py  # TFB Llama (vLLM)
│   │       └── tfb_qwen2_vllm.py  # TFB Qwen2 (vLLM)
│   └── setup.py
├── bash_scripts/                  # Shell scripts for experiments
│   ├── unc_greedy_single_para_batch.sh  # Single greedy generation
│   ├── unc_greedy_para.sh               # Multi-particle generation
│   ├── eval_detect.sh                   # Detection evaluation
│   └── run_scaling_multi_gpu.sh         # Test-time scaling evaluation
├── datasets/                      # Dataset download utilities
│   └── download_data.py
├── eval/                          # Evaluation scripts
│   ├── eval_detect_multi_seed.py  # Multi-seed detection evaluation
│   └── eval_scaling_test_multi_gpu.py  # Multi-GPU scaling evaluation (Table 2)
├── run/                           # Inference scripts
│   ├── greedy_unc_single_batch_refine.py  # Batch greedy + uncertainty
│   ├── greedy_responses_unc.py            # Per-sample uncertainty
│   └── utils/                             # Shared utilities
│       ├── config.py              # Experiment configuration
│       ├── grader.py              # Math answer grading
│       ├── math.py                # Math aggregation utilities
│       └── qwen_math_parser.py    # Answer extraction & parsing
├── convert_to_tfb.py              # Convert base models to TFB format
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{TokUR,
  title={TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning},
  author={Zhang, Tunyu and Shi, Haizhou and Wang, Yibin and Wang, Hengyi and He, Xiaoxiao and Li, Zhuowei and Chen, Haoxian and Han, Ligong and Xu, Kai and Zhang, Huan and Metaxas, Dimitris and Wang, Hao},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## Acknowledgements

This work builds on several open-source projects:
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for model infrastructure
- [BLoB](https://github.com/Wang-ML-Lab/bayesian-peft) and [TFB](https://github.com/haizhou-shi/bayesian-lm) for Bayesian LLM foundations

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
