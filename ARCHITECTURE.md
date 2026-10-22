# Architecture & Technical Details

## Intermediate Layer Ensembling (ILE) Framework

This document outlines the architectural implementation and modular design of the **Intermediate Layer Ensembling (ILE)** framework for sample-free epistemic uncertainty quantification and uncertainty decomposition in Large Language Models.

---

## 1. System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Intermediate Layer Ensembling (ILE)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: Salient Knowledge Token Identification                        │
│     ├─ compute_token_score_ranking() [src/token_attribution.py]         │
│     ├─ Attribution scoring via DeBERTa NLI embedding autograd           │
│     │  (Adapted from SDLG; Aichberger et al., ICLR 2025)                │
│     └─ Filters answer-bearing positions K without syntax dilution       │
│                                                                         │
│  Stage 2: In-Situ Intermediate Representation Probing                   │
│     ├─ generate_layer_ensemble_sequences() [src/layer_ensemble.py]      │
│     ├─ Extract hidden states across network depth j in {1, ..., L-1}    │
│     └─ Project via frozen LM head onto vocabulary support: q_{j,k}      │
│                                                                         │
│  Stage 3: Vectorized Disagreement & Uncertainty Decomposition           │
│     ├─ compute_all_layers_disagreement_vectorized() [src/disagree.py]   │
│     ├─ Total Uncertainty (TU)      = Terminal predictive entropy        │
│     ├─ Aleatoric Uncertainty (AU)  = Shallow mixture entropy            │
│     └─ Epistemic Uncertainty (EU)  = Deep representational drift / JSD  │
│                                                                         │
│  Stage 4: Downstream Evaluation & Selective Decision Routing            │
│     ├─ Benchmark verification [src/experiments/]                        │
│     ├─ AUROC, AURC, and PRR analysis [src/analysis/]                    │
│     └─ Selective RAG routing case study [notebooks/rag_routing_*.ipynb] │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Responsibilities

### `src/token_attribution.py`
- **Role**: Identifies high-salience knowledge tokens $\mathcal{K}$ on greedy generated sequences to focus uncertainty estimation on answer-bearing positions.
- **Attribution Source**: Adapted from the token attribution scoring algorithm of SDLG ([Aichberger et al., ICLR 2025](https://openreview.net/forum?id=HSi4VetQLj)).
- **Key Functions**:
  - `compute_token_score_ranking()`: Computes combined ranking $\mathcal{S}(k, v) = \alpha_1 \mathcal{R}_{\mathrm{attr}} + \alpha_2 \mathcal{R}_{\mathrm{subst}} + \alpha_3 \mathcal{R}_{\mathrm{imp}}$ via direct autograd on DeBERTa input embeddings.
  - `get_word_indices()`: Maps word boundary spans to model token subwords.
  - `rank_tensor()`: Rank calculation handling ties via unique consecutive values.

### `src/layer_ensemble.py`
- **Role**: Core generation and layer-wise ensembling engine of the ILE framework.
- **Key Functions**:
  - `generate_layer_ensemble_sequences()`: Performs candidate generation and probes intermediate hidden representations.
  - `get_model_layer_count()`: Inspects transformer config across LLaMA, Qwen, Phi, and MoE architectures to determine total depth $L$.

### `src/layer_ensemble_var.py`
- **Role**: Multi-range cumulative layer ensembling variant.
- **Key Functions**:
  - `generate_layer_ensemble_var_sequences()`: Evaluates cumulative layer ranges $[L-2, L-1]$ down to $[0, \dots, L-1]$, formatting outputs for multi-depth analysis.

### `src/disagree.py`
- **Role**: Vectorized divergence computation suite.
- **Key Functions**:
  - `compute_all_layers_disagreement_vectorized()`: Batched computation of Jensen-Shannon divergence, Entropy Mixture, and Gated MI across all depths in a single GPU call.
  - `calculate_all_disagreement_metrics()`: Comprehensive evaluation of 11 divergence metrics (JSD, KL, TVD, Bhattacharyya, EPKL, CV, etc.).
  - `get_relative_top_filter()`: Relative top probability filtering to eliminate uncalibrated vocabulary tail noise.

### `src/utils.py`
- **Role**: Model initializers, seed management, evaluation metrics, and semantic clustering utilities.
- **Key Functions**:
  - `get_models_and_tokenizers()`: Unified loader supporting 4-bit/8-bit BitsAndBytes quantization and FlashAttention-2.
  - `compute_correctness()`: BLEURT (`bleurt-base-128`) and task-specific string match scoring.
  - `compute_semantic_clusters()`: DeBERTa cross-encoder semantic clustering for Semantic Entropy baselines.

### `src/args.py`
- **Role**: Dataclass-based hyperparameter configuration.
- **Parameters**: Model identifiers, quantization levels, attribution weights ($\alpha_1, \alpha_2, \alpha_3$), and layer depth ratios.

---

## 3. Mathematical Foundations

Given prompt $\bm{x}$ and generated response $\bm{y} = (y_1, \ldots, y_N)$, ILE evaluates predictive reliability by contrasting terminal and intermediate representations:

1. **Terminal Vocabulary Distribution**:
   $$p_k = \mathrm{softmax}\big(\phi(\bm{h}_k^{(L)})\big)$$
2. **Intermediate Layer Distribution**:
   $$q_{j,k} = \mathrm{softmax}\big(\phi(\bm{h}_k^{(j)})\big), \quad j < L$$
3. **Layer Mixture Distribution**:
   $$\bar{m}_k^{(j)} = \frac{1}{2}\big(p_k + q_{j,k}\big)$$
4. **Analytical Drift Decomposition**:
   $$\Delta \mathrm{H}_j = \mathrm{H}(\bar{m}_k^{(j)}) - \frac{1}{2}\mathrm{H}(p_k) = \mathrm{JSD}(p_k \,\|\, q_{j,k}) + \frac{1}{2}\mathrm{H}(q_{j,k})$$

The first term isolates directional representational conflict ($\mathrm{JSD}$ / Epistemic Uncertainty), while terminal entropy $\mathrm{H}(p_k)$ captures Total Uncertainty.
