"""Intermediate Layer Ensembling (ILE) Analysis Module.

Provides evaluation routines, decomposition metrics, AUROC/AURC calculations,
and diagnostic visualization tools for epistemic uncertainty quantification.
"""

from .decomposition_metrics import compute_1d_metrics

__all__ = ["compute_1d_metrics"]
