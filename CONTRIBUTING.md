# Contributing Guidelines

## Development Workflow

### Setting Up Environment

```bash
# Clone the repository
git clone https://github.com/your-username/intermediate-layer-ensembling.git
cd intermediate-layer-ensembling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments and Analysis

Scripts are executed directly using standard Python commands with no package installation required. Benchmarks load datasets directly from Hugging Face:

```bash
# Run individual experiment pipelines
python src/experiments/run_experiment_truthfulqa.py
python src/experiments/run_experiment_scienceqa.py
python src/experiments/run_experiment_openbookqa.py

# Run all-layers in-situ uncertainty probing
python src/experiments/run_all_layers_efficient.py

# Run selective RAG case study
python src/experiments/run_rag_openbookqa.py
```

### Code Organization

- **src/**: Core Python modules (`layer_ensemble.py`, `layer_ensemble_var.py`, `token_attribution.py`, `disagree.py`, `args.py`, `utils.py`)
- **src/experiments/**: Benchmark runners and baseline implementations (ICE, TokUR)
- **src/analysis/**: Evaluation metrics and decomposition algorithms
- **notebooks/**: Jupyter notebooks for paper analysis and visualization
- **configs/**: Experiment parameter configurations

### Code Standards

1. **Formatting**: Format Python files with standard PEP 8 formatting (`black src/`).
2. **Reproducibility**: Ensure all experiment runners allow setting seeds via `seed_everything(args.seed_value)` and use dynamic relative path resolution.
3. **No Hardcoded Tokens or Paths**: Never commit API keys or absolute filesystem paths. Read credentials from environment variables (e.g. `HF_TOKEN`).
4. **Attribution Integrity**: Maintain clear distinction between the core Intermediate Layer Ensembling framework and external baselines or attribution components (e.g. SDLG token attribution).
