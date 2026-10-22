"""Backward-compatibility shim for SDLG imports.

The core implementation has been modularized under:
- `layer_ensemble.py`: Intermediate Layer Ensembling (ILE) framework.
- `token_attribution.py`: Salient knowledge token identification (attribution scoring adapted from Aichberger et al., 2025).
"""

from .token_attribution import (
    forward_hook,
    get_word_indices,
    rank_tensor,
    compute_token_score_ranking,
)
from .layer_ensemble import (
    get_model_layer_count,
    generate_layer_ensemble_sequences,
    generate_semantically_diverse_output_sequences,
)

__all__ = [
    "forward_hook",
    "get_word_indices",
    "rank_tensor",
    "compute_token_score_ranking",
    "get_model_layer_count",
    "generate_layer_ensemble_sequences",
    "generate_semantically_diverse_output_sequences",
]
