# Comprehensive Guide to Evaluation Metrics in Selective RAG Routing

This document provides complete, publication-grade mathematical definitions, empirical motivations, and diagnostic interpretations for all evaluation metrics used in `rag_routing_analysis.ipynb`.

---

## 1. Ground Truth Correctness & Sample State Transitions

### 1.1 Binary Correctness Metric ($y_{\mathrm{direct}}, y_{\mathrm{rag}}$)

* **Definition**: A binary indicator denoting whether the model's generated answer correctly resolves the query.
* **Evaluation Standard**: Computed via the legacy lightweight `bleurt-base-128` checkpoint against official ground-truth references:
  $$y = \mathbb{I}\left(\mathrm{BLEURT}_{\text{base-128}}(\hat{a}, a^*) \ge 0.0\right)$$
  where $\hat{a}$ is the greedy decoded prediction and $a^*$ is the reference target.
* **Benchmark Invariance**: The correctness threshold is permanently anchored at **0.0** across all evaluations.

### 1.2 Net Accuracy Shift ($\Delta \mathrm{Acc}$)

* **Definition**: The net percentage-point improvement in sample-level accuracy upon external context integration:
  $$\Delta \mathrm{Acc} = \frac{1}{N} \sum_{i=1}^N \left( y_{\mathrm{rag}}^{(i)} - y_{\mathrm{direct}}^{(i)} \right) \times 100\% = \mathrm{Acc}_{\mathrm{rag}} - \mathrm{Acc}_{\mathrm{direct}}$$
* **Interpretation**:
  - Positive values indicate that external retrieval provided a net benefit across the dataset.
  - Negative values signify that in-context distraction corrupted more correct answers than retrieval repaired failing queries.

### 1.3 Correctness State Transition Taxonomy

Every sample $i \in \{1, \dots, N\}$ is partitioned into one of four mutually exclusive state transitions:

| State Transition | Binary Shift | Symbol | Formal Definition | Semantic Interpretation |
| :--- | :---: | :---: | :--- | :--- |
| **Recovered** | $0 \rightarrow 1$ | $N_{0 \rightarrow 1}$ | $\sum_{i=1}^N \mathbb{I}(y_{\mathrm{direct}}^{(i)} = 0 \land y_{\mathrm{rag}}^{(i)} = 1)$ | Internal parametric knowledge failure successfully resolved by external context. |
| **Retained** | $1 \rightarrow 1$ | $N_{1 \rightarrow 1}$ | $\sum_{i=1}^N \mathbb{I}(y_{\mathrm{direct}}^{(i)} = 1 \land y_{\mathrm{rag}}^{(i)} = 1)$ | Existing parametric knowledge correctly preserved despite external context insertion. |
| **Distracted** | $1 \rightarrow 0$ | $N_{1 \rightarrow 0}$ | $\sum_{i=1}^N \mathbb{I}(y_{\mathrm{direct}}^{(i)} = 1 \land y_{\mathrm{rag}}^{(i)} = 0)$ | Correct direct answer corrupted by misleading or irrelevant in-context distractor noise. |
| **Persistent** | $0 \rightarrow 0$ | $N_{0 \rightarrow 0}$ | $\sum_{i=1}^N \mathbb{I}(y_{\mathrm{direct}}^{(i)} = 0 \land y_{\mathrm{rag}}^{(i)} = 0)$ | Unresolved knowledge deficit where neither internal weights nor external passages resolved the query. |

---

## 2. Retrieval Diagnostics & Context Quality

### 2.1 Retriever Recall ($\mathrm{Hit}@K$)

* **Definition**: The fraction of queries where at least one of the top-$K$ retrieved passages contains the necessary gold supporting premise:
  $$\mathrm{Hit}@K = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \exists p \in \mathcal{P}_K^{(i)} : \mathrm{ContainsPremise}(p) \right)$$
* **Benchmark Characteristics on OpenBookQA**:
  - **Dense Top-1 ($K=1$)**: $\mathrm{Hit}@1 = 52.0\%$ (Strict semantic relevance, maximum premise density).
  - **Dense Top-3 ($K=3$)**: $\mathrm{Hit}@3 = 71.8\%$ (High recall, but introduces $2 \times$ distractor noise).
  - **BM25+ Top-3 ($K=3$)**: $\mathrm{Hit}@3 = 57.4\%$ (Distilled lexical retrieval with optimized length normalization).
  - **BM25 Raw Top-3 ($K=3$)**: $\mathrm{Hit}@3 = 39.0\%$ (Unoptimized lexical baseline).

### 2.2 Deficit Recovery Rate ($\mathrm{DRR}$)

* **Definition**: The proportion of initially failing queries that are successfully corrected by retrieval:
  $$\mathrm{DRR} = \frac{N_{0 \rightarrow 1}}{N_{0 \rightarrow 1} + N_{0 \rightarrow 0}} = \frac{N_{0 \rightarrow 1}}{N_{\text{incorrect, direct}}}$$
* **Diagnostic Purpose**: Isolates how effectively the model exploits injected evidence when it genuinely lacks parametric knowledge, independent of baseline parametric strength.

### 2.3 Distraction Rate ($\mathrm{DR}$)

* **Definition**: The proportion of initially correct answers corrupted into incorrect outputs by retrieval context:
  $$\mathrm{DR} = \frac{N_{1 \rightarrow 0}}{N_{1 \rightarrow 1} + N_{1 \rightarrow 0}} = \frac{N_{1 \rightarrow 0}}{N_{\text{correct, direct}}}$$
* **Diagnostic Purpose**: Quantifies the vulnerability of the architecture to distractor interference and semantic context corruption.

---

## 3. Layer-Ensemble Uncertainty Decomposition

### 3.1 Predictive Entropy and Layer Mixtures

Given an autoregressive decoder with $L$ layers, let $p_l \in \Delta^{|V|}$ denote the vocabulary projection (via unembedding head) of the hidden representations at layer $l \in \{1, \dots, L\}$.

* **Final Layer Predictive Entropy**:
  $$\mathcal{H}(p_L) = - \sum_{v \in V} p_L(v) \log p_L(v)$$

### 3.2 Epistemic Uncertainty ($\mathrm{EU}$)

* **Definition**: Epistemic uncertainty is quantified through late-layer ensemble divergence:
  $$\mathrm{EU} = \mathcal{H}(p_L) - \mathcal{H}\left( \sum_{l=K_e}^L w_l \, p_l \right)$$
  where $K_e = \min(27, L - 1)$ marks the late-stage representational boundary and $w_l$ represents uniform ensembling weights.
* **Physical Intuition**: Reflects model uncertainty arising from incomplete internal knowledge or representational instability across deep reasoning layers.

### 3.3 Aleatoric Uncertainty ($\mathrm{AU}$)

* **Definition**: Aleatoric uncertainty captures inherent query ambiguity via early layer representations:
  $$\mathrm{AU} = \mathcal{H}\left( \sum_{l=0}^{K_a} w_l \, p_l \right)$$
  where $K_a = \min(2, L - 1)$ captures shallow semantic and syntax encoding.
* **Physical Intuition**: Invariance to external retrieval; reflects linguistic ambiguity or underspecified question framing.

---

## 4. Epistemic Dissipation Dynamics

### 4.1 Uncertainty Dissipation ($\Delta \mathrm{EU}$)

* **Definition**: The reduction in epistemic uncertainty achieved by incorporating external retrieval context:
  $$\Delta \mathrm{EU} = \mathrm{EU}_{\mathrm{direct}} - \mathrm{EU}_{\mathrm{rag}}$$
* **Interpretation**:
  - $\Delta \mathrm{EU} > 0$: Successful uncertainty dissipation (the external context resolved internal ambiguity).
  - $\Delta \mathrm{EU} \le 0$: Stagnant or exacerbated uncertainty (the external context added distractor noise or conflicting evidence).

### 4.2 Epistemic Discrimination Margin

* **Definition**: The statistical contrast in uncertainty dissipation between retriever hits and retriever misses:
  $$\text{Margin} = \mu\left(\Delta \mathrm{EU} \mid \mathrm{Hit}@K = 1\right) - \mu\left(\Delta \mathrm{EU} \mid \mathrm{Hit}@K = 0\right)$$
* **Theoretical Implication**: A positive margin confirms that the internal layer ensemble faithfully discriminates true external premises from misleading distractor noise without external verification labels.

### 4.3 Epistemic Grounding Contrast ($\Gamma_{\mathrm{EU}}$)

* **Definition**: The contrast in uncertainty dissipation between successful recoveries ($0 \rightarrow 1$) and distraction failures ($1 \rightarrow 0$):
  $$\Gamma_{\mathrm{EU}} = \mu\left(\Delta \mathrm{EU}_{0 \rightarrow 1}\right) - \mu\left(\Delta \mathrm{EU}_{1 \rightarrow 0}\right)$$
* **Theoretical Implication**: Measures whether uncertainty reduction correlates with true factual correction rather than ungrounded model hallucination.

---

## 5. Budget-Constrained Selective Routing & Pareto Efficiency

### 5.1 Invocation Budget ($\beta$)

* **Definition**: The fraction of total queries for which external retrieval augmentation is invoked:
  $$\beta \in [0.0, 1.0]$$
  - $\beta = 0.0$: Pure Direct Generation (Zero retrieval cost, minimal prompt latency).
  - $\beta = 1.0$: Full Unselective RAG (100% retrieval and context-processing cost).

### 5.2 Routing Policy by Epistemic Priority

Queries are ranked in descending order of direct epistemic uncertainty:
$$\pi = \mathrm{argsort}\left( -\mathrm{EU}_{\mathrm{direct}} \right)$$
For a given budget fraction $\beta$, the top $k = \lfloor \beta \cdot N \rfloor$ most uncertain queries trigger retrieval:
$$y_{\mathrm{routed}}^{(i)}(\beta) = \begin{cases} y_{\mathrm{rag}}^{(i)}, & \text{if } i \in \pi[:k] \\ y_{\mathrm{direct}}^{(i)}, & \text{otherwise} \end{cases}$$

### 5.3 Fixed-Budget Performance Metrics

* **`Acc @ 25% Budget` & `Gain @ 25%`**:
  $$\mathrm{Acc}_{25\%} = \frac{1}{N} \sum_{i=1}^N y_{\mathrm{routed}}^{(i)}(0.25), \quad \mathrm{Gain}_{25\%} = \mathrm{Acc}_{25\%} - \mathrm{Acc}_{\mathrm{direct}}$$
  Quantifies accuracy unlocked when saving **75%** of retrieval calls and prompt context expansion.
* **`Acc @ 50% Budget` & `Gain @ 50%`**:
  $$\mathrm{Acc}_{50\%} = \frac{1}{N} \sum_{i=1}^N y_{\mathrm{routed}}^{(i)}(0.50), \quad \mathrm{Gain}_{50\%} = \mathrm{Acc}_{50\%} - \mathrm{Acc}_{\mathrm{direct}}$$
  Measures the balanced compute operating point.

### 5.4 Peak Routed Accuracy & Optimal Budget ($\beta^*$)

* **Definition**: The global maximum of the empirical accuracy frontier across all budget allocations:
  $$\mathrm{Acc}^* = \max_{\beta \in [0, 1]} \mathrm{Acc}_{\mathrm{routed}}(\beta), \quad \beta^* = \arg\max_{\beta \in [0, 1]} \mathrm{Acc}_{\mathrm{routed}}(\beta)$$
* **Selective Routing Gain**:
  $$\mathrm{SRG} = \mathrm{Acc}^* - \mathrm{Acc}_{\mathrm{rag}}$$
  The accuracy improvement achieved by selectively routing at $\beta^*$ compared against indiscriminate 100% RAG. When positive, selective routing strictly outperforms full RAG while requiring fewer compute resources.

### 5.5 Distraction Shielding Rate ($\mathrm{DSR}$)

* **Definition**: The percentage of potential distraction errors averted by routing queries at the optimal budget $\beta^*$ instead of passing 100% of samples through retrieval:
  $$\mathrm{DSR} = \frac{N_{1 \rightarrow 0} - N_{1 \rightarrow 0}^{\text{routed}}(\beta^*)}{N_{1 \rightarrow 0}} \times 100\%$$
* **Semantic Meaning**: Quantifies the protective capability of epistemic gating against contextual noise.

### 5.6 Routed Recovery/Distraction Ratio

* **Definition**: The ratio of recovered correct answers to admitted distraction errors under the routed subset:
  $$\text{Ratio} = \frac{N_{0 \rightarrow 1}^{\text{routed}}(\beta^*)}{\max\left(1, N_{1 \rightarrow 0}^{\text{routed}}(\beta^*)\right)}$$
* **Interpretation**: Values strictly greater than $1.0\times$ verify that the router admitted more beneficial knowledge updates than destructive distractor errors.

---

## 6. Targeted Ablation Metrics

### 6.1 Density Gain (Table A)

* **Definition**: The net accuracy margin between Dense Top-1 and Dense Top-3 under full RAG:
  $$\text{Density Gain} = \mathrm{Acc}_{\mathrm{rag}}(\text{Top-1}) - \mathrm{Acc}_{\mathrm{rag}}(\text{Top-3})$$
* **Purpose**: Evaluates whether compressing retrieved context to the single highest-scoring passage improves generation by eliminating distractor noise.

### 6.2 Distraction Reduction (Table A)

* **Definition**: The percentage-point drop in distraction rate when restricting context from Top-3 to Top-1:
  $$\text{Distraction Reduction} = \mathrm{DR}_{\text{Top-3}} - \mathrm{DR}_{\text{Top-1}}$$
* **Purpose**: Directly verifies whether in-context distractor interference is mitigated by context density optimization.

### 6.3 Modality Grounding Ratio (Table B)

* **Definition**: Comparative evaluation of recovery rates between dense semantic representations ($\mathrm{DRR}_{\mathrm{Dense}}$) and BM25+ distilled lexical representations ($\mathrm{DRR}_{\mathrm{BM25+}}$):
  $$\text{Modality Contrast} = \mathrm{DRR}_{\mathrm{Dense}} - \mathrm{DRR}_{\mathrm{BM25+}}$$
* **Purpose**: Tests whether dense embeddings or keyword-focused lexical passages provide superior factual grounding when retrieved.
