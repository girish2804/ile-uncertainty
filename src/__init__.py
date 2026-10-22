"""Intermediate Layer Ensembling (ILE) Framework.

Uncertainty Decomposition for Large Language Models via Intermediate Layer Ensembling.

Core Python package providing sample-free epistemic uncertainty quantification,
layer disagreement probing, and salient knowledge token attribution.
"""

__version__ = "1.0.0"

from .token_attribution import (
    compute_token_score_ranking,
    rank_tensor,
    get_word_indices,
    forward_hook,
)
from .layer_ensemble import (
    generate_layer_ensemble_sequences,
    generate_semantically_diverse_output_sequences,
    get_model_layer_count,
)
from .disagree import (
    calculate_all_disagreement_metrics,
    compute_all_layers_disagreement_vectorized,
    get_relative_top_filter,
)
from .utils import (
    seed_everything,
    get_models_and_tokenizers,
    generate_text,
    compute_correctness,
    compute_likelihood,
    compute_semantic_entropy,
    compute_semantic_clusters,
    compute_batched_semantic_pairs,
)

__all__ = [
    "compute_token_score_ranking",
    "generate_layer_ensemble_sequences",
    "generate_semantically_diverse_output_sequences",
    "get_model_layer_count",
    "rank_tensor",
    "get_word_indices",
    "forward_hook",
    "calculate_all_disagreement_metrics",
    "compute_all_layers_disagreement_vectorized",
    "get_relative_top_filter",
    "seed_everything",
    "get_models_and_tokenizers",
    "generate_text",
    "compute_correctness",
    "compute_likelihood",
    "compute_semantic_entropy",
    "compute_semantic_clusters",
    "compute_batched_semantic_pairs",
]
