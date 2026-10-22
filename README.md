# Intermediate Layer Ensembling (ILE): Uncertainty Decomposition for Large Language Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **Intermediate Layer Ensembling (ILE)**: A lightweight, sample-free framework for epistemic uncertainty quantification and uncertainty decomposition in autoregressive Large Language Models.

---

## Overview

Quantifying predictive uncertainty in autoregressive language models is critical for reliable deployment in high-stakes reasoning, question answering, and decision-making systems. While existing methods rely on expensive repeated stochastic sampling or external weight perturbations, **Intermediate Layer Ensembling (ILE)** probes the hierarchical representational dynamics across transformer network depth in a **single forward pass**.

### Key Contributions & Pipeline

1. **Sample-Free Epistemic Uncertainty Quantification**: Probes intermediate transformer hidden states via the frozen language model head onto a shared vocabulary distribution, extracting depth-wise disagreement without repeated sampling.
2. **Analytical Uncertainty Decomposition**: Decomposes Total Uncertainty ($\mathrm{TU}$) into Aleatoric Uncertainty ($\mathrm{AU}$) via shallow layer mixtures and Epistemic Uncertainty ($\mathrm{EU}$) via deep representational drift (Jensen-Shannon Divergence, EPKL, and Gated Mutual Information).
3. **Salient Knowledge Token Selection (SDLG Attribution)**: Uniform averaging across the full response dilutes factual error signals with non-informative syntax and stop words. We focus uncertainty evaluation on answer-bearing positions identified via the gradient attribution scoring formulation adapted from SDLG ([Aichberger et al., ICLR 2025](https://openreview.net/forum?id=HSi4VetQLj)).
4. **Vectorized Divergence Suite**: Vectorized computation of 11 divergence and epistemic uncertainty metrics (JSD, KL, TVD, Bhattacharyya, EPKL, CV, Entropy Mixture, Gated MI) across all intermediate layer depths $k \in \{1, \dots, L-1\}$.
5. **Selective RAG Decision Routing**: Empirical validation demonstrating that external retrieval selectively dissipates epistemic uncertainty on factual knowledge failures, establishing Pareto-optimal accuracy-efficiency frontiers.

---

## Repository Structure

```
ile/
├── src/                                # Core Python source modules
│   ├── __init__.py                     # Package interface
│   ├── args.py                         # Configuration dataclass and parameter management
│   ├── layer_ensemble.py               # Intermediate Layer Ensembling (ILE) framework
│   ├── layer_ensemble_var.py           # Multi-range cumulative layer ensembling variant
│   ├── token_attribution.py            # Salient token identification (adapted from SDLG)
│   ├── disagree.py                     # Disagreement metrics and vectorized layer ensembling
│   ├── utils.py                        # Model loading, correctness evaluation, and clustering
│   ├── sdlg.py                         # Backward-compatibility import shim
│   ├── sdlg_var.py                     # Backward-compatibility import shim
│   ├── analysis/                       # Evaluation and decomposition metrics
│   │   ├── decomposition_metrics.py    # AUROC, AURC, and 1D/2D rejection metrics
│   │   ├── run_decomposition_eval.py   # Decomposition evaluation pipeline
│   │   └── auroc_delta_viz.py          # Visualization routines
│   └── experiments/                    # Benchmark experiment runners and baselines
│       ├── run_experiment_truthfulqa.py # TruthfulQA evaluation pipeline
│       ├── run_experiment_scienceqa.py  # ScienceQA evaluation pipeline
│       ├── run_experiment_openbookqa.py # OpenBookQA evaluation pipeline
│       ├── run_all_layers_efficient.py # Sampling-free multi-layer uncertainty probing
│       ├── run_rag_openbookqa.py       # Selective RAG routing case study
│       ├── benchmark_computational_cost.py # GPU latency and memory benchmarking
│       ├── ice/                        # Input Clarification Ensembling baseline
│       ├── tokur/                      # TokUR weight perturbation baseline
│       └── legacy/                     # Reference experiment implementations
├── notebooks/                          # Interactive analysis and paper figure generation
│   ├── paper_layerwise_plots.ipynb     # Main layerwise uncertainty visualizations (Fig. 1 & 2)
│   ├── auroc_analysis.ipynb            # AUROC benchmark comparisons (Table 1)
│   ├── decomposition_analysis.ipynb    # Symmetric uncertainty decomposition (Ka = Ke)
│   ├── asymmetric_decomposition_analysis.ipynb # Asymmetric decomposition (Ka = 2, Ke = 16)
│   ├── difference_decomposition_analysis.ipynb # Difference decomposition & architecture blocks
│   ├── rag_routing_analysis.ipynb      # Selective RAG case study under budget constraints
│   ├── threshold_sweep_analysis.ipynb  # Filtering threshold sensitivity analysis
│   └── appendix_qualitative_analysis.ipynb # Qualitative case studies
├── configs/                            # Experiment configuration files
├── result/                             # Output directory for experimental runs and benchmarks
├── requirements.txt                    # Project dependencies
├── ARCHITECTURE.md                     # Technical architecture documentation
├── METRICS_GUIDE.md                    # Mathematical definitions of evaluation metrics
├── CONTRIBUTING.md                     # Development workflow guidelines
└── LICENSE                             # MIT License
```

---

## Installation

### Prerequisites
- Python $\ge$ 3.9
- CUDA-enabled GPU (NVIDIA RTX 3090, A100, A6000, or H100 recommended for 70B models)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/intermediate-layer-ensembling.git
cd intermediate-layer-ensembling

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch matching your CUDA version (e.g., CUDA 11.8 or 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Hugging Face Authentication
For gated models (e.g., Llama 3, Gemma), set your Hugging Face access token:
```bash
export HF_TOKEN="your_huggingface_token"
# or run:
huggingface-cli login
```

Datasets (OpenBookQA, ScienceQA, TruthfulQA) are downloaded automatically from Hugging Face on first execution.

---

## Running Experiments

Run the experiment runners one by one using standard Python commands. All outputs and checkpoint files are written directly into `result/`:

### 1. Main Benchmark Runs
```bash
# TruthfulQA benchmark
python src/experiments/run_experiment_truthfulqa.py

# ScienceQA benchmark
python src/experiments/run_experiment_scienceqa.py

# OpenBookQA benchmark
python src/experiments/run_experiment_openbookqa.py
```

### 2. Multi-Layer Uncertainty Probing (Sampling-Free)
```bash
# Evaluates layerwise epistemic divergence across all intermediate depths
python src/experiments/run_all_layers_efficient.py
```

### 3. Selective RAG Case Study
```bash
# Evaluates selective retrieval decisions under varying uncertainty thresholds
python src/experiments/run_rag_openbookqa.py
```

### 4. Baseline Comparisons
```bash
# Run TokUR stochastic weight perturbation baseline
python src/experiments/tokur/run_tokur_baseline.py

# Run ICE (Input Clarification Ensembling) baseline
python src/experiments/ice/run_ice_baseline.py
```

### 5. Computational Cost Benchmarking
```bash
# Measure latency, peak memory, and throughput
python src/experiments/benchmark_computational_cost.py
```

---

## Analysis & Paper Notebooks

Launch Jupyter to explore the analysis notebooks in `notebooks/`. Each notebook corresponds directly to results presented in the paper:

```bash
jupyter notebook notebooks/
```

| Notebook | Paper Results | Description |
|:---|:---|:---|
| [`paper_layerwise_plots.ipynb`](notebooks/paper_layerwise_plots.ipynb) | **Figure 3** | Layerwise epistemic divergence patterns across model architectures ($3 \times 3$ and $3 \times 6$ grids). |
| [`auroc_analysis.ipynb`](notebooks/auroc_analysis.ipynb) | **Section 4.4** | Main AUROC error-detection and PRR benchmark comparisons across all models and datasets. |
| [`difference_decomposition_analysis.ipynb`](notebooks/difference_decomposition_analysis.ipynb) | **Table 1** | Difference decomposition metrics across architecture classes (Llama, Qwen, Phi). |
| [`rag_routing_analysis.ipynb`](notebooks/rag_routing_analysis.ipynb) | **Table 2** | Selective RAG case study: epistemic uncertainty dissipation and accuracy-retrieval tradeoffs. |

---

## Supported Models & Datasets

### Models
- **Meta Llama**: `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`
- **Qwen 2.5**: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`
- **Microsoft Phi-3**: `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-medium-4k-instruct`

### Datasets
- **OpenBookQA**: Elementary science question answering assessing retrieval and commonsense reasoning.
- **ScienceQA**: Multi-choice scientific reasoning.
- **TruthfulQA**: Evaluation of truthfulness and hallucination rates across open domains.

---

## Attribution & Related Work

- **Salient Token Attribution**: Salient knowledge token identification in [`src/token_attribution.py`](src/token_attribution.py) is adapted from the attribution scoring formulation in:
  > Lukas Aichberger, Kajetan Schweighofer, Mykyta Ielanskyi, and Sepp Hochreiter. *Improving Uncertainty Estimation through Semantically Diverse Language Generation (SDLG)*. International Conference on Learning Representations (ICLR), 2025.
- **Baseline Implementations**: Comparisons include Semantic Entropy ([Kuhn et al., 2023](https://arxiv.org/abs/2302.09664)), TokUR ([Du et al., 2024](https://arxiv.org/abs/2406.14175)), and ICE ([Zhang et al., 2024](https://arxiv.org/abs/2406.15927)).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
