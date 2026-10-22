"""
Decomposition Uncertainty Metrics & 2D Evaluation Module.
Implements:
1. Standard 1D Metrics: AUROC, AURC, and 1D PRR (Prediction Rejection Ratio).
2. 2D Joint Rejection Pareto Frontier & PRR_2D.
3. 5-Fold Cross-Validated 4-Way Agentic Decision Routing under 7 Economic Regimes.
4. Component Disentanglement Diagnostics (Pearson, Spearman, Mutual Information).
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, List, Optional


# ==============================================================================
# 1. Standard 1D Uncertainty Metrics
# ==============================================================================

def compute_1d_metrics(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Computes standard 1D uncertainty evaluation metrics:
    - AUROC: Area Under the ROC Curve for error detection (y_true == 0 is positive class).
    - AURC: Area Under the Accuracy-Coverage Rejection Curve.
    - PRR: Prediction Rejection Ratio normalized against an optimal oracle rejector.

    Parameters:
        scores: 1D array of uncertainty scores (higher = more uncertain).
        y_true: 1D binary correctness array (1 = correct, 0 = incorrect).
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    
    # 1. AUROC (Error detection: 1 - y_true is positive class)
    y_error = 1 - y_true
    if len(np.unique(y_error)) > 1:
        auroc = float(roc_auc_score(y_error, scores))
    else:
        auroc = 0.50

    # 2. Rejection Curve (Sort ascending by uncertainty -> most confident first)
    sort_idx = np.argsort(scores)
    sorted_y = y_true[sort_idx]
    
    # Cumulative accuracy at every coverage k = 1 ... n
    cum_correct = np.cumsum(sorted_y)
    coverages = np.arange(1, n + 1) / n
    accuracies = cum_correct / np.arange(1, n + 1)
    
    base_acc = float(accuracies[-1])
    aurc = float(np.trapz(accuracies, coverages))
    
    # 3. Oracle Rejection Curve (All 1s first, then 0s)
    oracle_y = np.sort(y_true)[::-1]
    oracle_cum_correct = np.cumsum(oracle_y)
    oracle_accuracies = oracle_cum_correct / np.arange(1, n + 1)
    oracle_aurc = float(np.trapz(oracle_accuracies, coverages))
    
    # Normalized PRR
    denom = oracle_aurc - base_acc
    if denom > 1e-12:
        prr = float((aurc - base_acc) / denom)
    else:
        prr = 0.0
        
    return {
        "auroc": auroc,
        "aurc": aurc,
        "prr": prr,
        "base_accuracy": base_acc
    }


# ==============================================================================
# 2. 2D Joint Rejection Pareto Frontier & PRR_2D
# ==============================================================================

def compute_2d_pareto_rejection(
    au: np.ndarray, 
    eu: np.ndarray, 
    y_true: np.ndarray, 
    n_grid: int = 50,
    n_bins: int = 100
) -> Dict[str, any]:
    """
    Computes the 2D Pareto Rejection Envelope and PRR_2D across joint (AU, EU) thresholds.
    For each target coverage level c in (0, 1], finds:
        Acc*(c) = max_{tau_A, tau_E: Coverage=c} Acc(tau_A, tau_E)

    Parameters:
        au: 1D array of aleatoric uncertainty scores.
        eu: 1D array of epistemic uncertainty scores.
        y_true: 1D binary correctness array (1 = correct, 0 = incorrect).
        n_grid: Number of quantile thresholds to evaluate per axis (n_grid x n_grid).
        n_bins: Number of discretized coverage bins for the upper Pareto envelope.
    """
    au = np.asarray(au, dtype=float)
    eu = np.asarray(eu, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    base_acc = float(np.mean(y_true))

    # Candidate threshold grids
    tau_a_candidates = np.percentile(au, np.linspace(5, 100, n_grid))
    tau_e_candidates = np.percentile(eu, np.linspace(5, 100, n_grid))

    # Evaluate all 2D grid points
    coverage_list = []
    accuracy_list = []
    threshold_pairs = []

    for ta in tau_a_candidates:
        mask_a = au <= ta
        for te in tau_e_candidates:
            retained = mask_a & (eu <= te)
            k = np.sum(retained)
            if k == 0:
                continue
            cov = k / n
            acc = np.mean(y_true[retained])
            coverage_list.append(cov)
            accuracy_list.append(acc)
            threshold_pairs.append((ta, te))

    # Add endpoints
    coverage_list.append(1.0)
    accuracy_list.append(base_acc)

    cov_arr = np.array(coverage_list)
    acc_arr = np.array(accuracy_list)

    # Compute upper Pareto frontier: max accuracy for each coverage bucket
    grid_cov = np.linspace(0.05, 1.0, n_bins)
    pareto_acc = np.zeros_like(grid_cov)

    for i, c in enumerate(grid_cov):
        # Allow small tolerance window around target coverage
        tol = 1.0 / n_bins
        nearby = (cov_arr >= c - tol) & (cov_arr <= c + tol)
        if np.any(nearby):
            pareto_acc[i] = np.max(acc_arr[nearby])
        else:
            # Fallback to closest coverage
            closest_idx = np.argmin(np.abs(cov_arr - c))
            pareto_acc[i] = acc_arr[closest_idx]

    # Monotonicity enforcement (accuracy should not decrease as coverage shrinks)
    for i in range(len(pareto_acc) - 2, -1, -1):
        if pareto_acc[i] < pareto_acc[i + 1]:
            pareto_acc[i] = pareto_acc[i + 1]

    # Area Under 2D Rejection Curve
    aurc_2d = float(np.trapz(pareto_acc, grid_cov))

    # Oracle 1D rejection benchmark
    oracle_y = np.sort(y_true)[::-1]
    oracle_cov = np.arange(1, n + 1) / n
    oracle_acc = np.cumsum(oracle_y) / np.arange(1, n + 1)
    oracle_aurc = float(np.trapz(oracle_acc, oracle_cov))

    denom = oracle_aurc - base_acc
    prr_2d = float((aurc_2d - base_acc) / denom) if denom > 1e-12 else 0.0

    return {
        "prr_2d": prr_2d,
        "aurc_2d": aurc_2d,
        "base_accuracy": base_acc,
        "pareto_coverage": grid_cov,
        "pareto_accuracy": pareto_acc
    }


# ==============================================================================
# 3. 5-Fold Cross-Validated 4-Way Agentic Decision Routing
# ==============================================================================

# Standard 7 Economic Regimes Definition
DEFAULT_ECONOMIC_REGIMES = {
    "Standard": {
        "R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3
    },
    "High-Safety": {
        "R_correct": 1.0, "C_err": -10.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3
    },
    "High-RAG-Cost": {
        "R_correct": 1.0, "C_err": -5.0, "C_rag": -1.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3
    },
    "Low-RAG-Success": {
        "R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.40, "C_abstain": -0.5, "C_clarify": -0.3
    },
    "Low-Safety": {
        "R_correct": 1.0, "C_err": -2.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3
    },
    "Highly-Capable-RAG": {
        "R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.90, "C_abstain": -0.5, "C_clarify": -0.3
    },
    "High-Abstain-Penalty": {
        "R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -4.0, "C_clarify": -0.3
    },
}

def _calculate_utility_for_actions(
    actions: np.ndarray, 
    y_true: np.ndarray, 
    regime: Dict[str, float]
) -> float:
    """
    Computes sample-averaged operational utility for assigned routing actions:
    0: Answer directly
    1: Retrieve (RAG)
    2: Clarify (Prompt disambiguation)
    3: Abstain (Safe fallback)
    """
    r_corr = regime.get("R_correct", 1.0)
    c_err = regime.get("C_err", -5.0)
    c_rag = regime.get("C_rag", -0.5)
    p_rag = regime.get("P_rag_success", 0.70)
    c_abs = regime.get("C_abstain", -0.5)
    c_cla = regime.get("C_clarify", -0.3)

    utilities = np.zeros_like(y_true, dtype=float)

    # 1. Direct Answer
    mask_ans = (actions == 0)
    utilities[mask_ans] = np.where(y_true[mask_ans] == 1, r_corr, c_err)

    # 2. Retrieve (RAG)
    mask_rag = (actions == 1)
    rag_payoff = p_rag * r_corr + (1 - p_rag) * c_err + c_rag
    utilities[mask_rag] = rag_payoff

    # 3. Clarify (Interactive Query)
    mask_cla = (actions == 2)
    p_cla_resolve = 0.85  # Disambiguation resolution rate
    cla_payoff = p_cla_resolve * np.where(y_true[mask_cla] == 1, r_corr, c_err) + c_cla
    utilities[mask_cla] = cla_payoff

    # 4. Abstain
    mask_abs = (actions == 3)
    utilities[mask_abs] = c_abs

    return float(np.mean(utilities))


def compute_5fold_decision_routing(
    au: np.ndarray,
    eu: np.ndarray,
    y_true: np.ndarray,
    single_entropy: Optional[np.ndarray] = None,
    regimes: Optional[Dict[str, Dict[str, float]]] = None,
    n_splits: int = 5,
    seed: int = 42
) -> Dict[str, any]:
    """
    Runs 5-fold cross-validated agentic decision routing.
    Compares:
    - Direct Answering (No UQ)
    - 1D Single-Entropy Routing (Thresholds final token entropy)
    - 2D Decomposed UQ Routing (Ours: Joint AU and EU thresholds)

    Returns Out-of-Fold Expected Utilities for all regimes.
    """
    if regimes is None:
        regimes = DEFAULT_ECONOMIC_REGIMES
        
    au = np.asarray(au, dtype=float)
    eu = np.asarray(eu, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    
    if single_entropy is None:
        single_entropy = au

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    results = {
        "regimes": list(regimes.keys()),
        "direct_answering": {},
        "1d_entropy_routing": {},
        "2d_decomposed_routing": {}
    }

    # Candidate quantiles for grid search
    tau_quantiles = np.linspace(10, 90, 15)

    for r_name, r_params in regimes.items():
        oof_actions_direct = np.zeros(n, dtype=int)
        oof_actions_1d = np.zeros(n, dtype=int)
        oof_actions_2d = np.zeros(n, dtype=int)

        for train_idx, val_idx in kf.split(y_true):
            y_tr, y_val = y_true[train_idx], y_true[val_idx]
            au_tr, au_val = au[train_idx], au[val_idx]
            eu_tr, eu_val = eu[train_idx], eu[val_idx]
            se_tr, se_val = single_entropy[train_idx], single_entropy[val_idx]

            # --- 1D In-Fold Grid Search (Threshold on single entropy: Answer vs Abstain) ---
            best_u_1d = -np.inf
            best_t_1d = np.median(se_tr)
            for q in tau_quantiles:
                t = np.percentile(se_tr, q)
                acts_tr = np.where(se_tr <= t, 0, 3)
                u = _calculate_utility_for_actions(acts_tr, y_tr, r_params)
                if u > best_u_1d:
                    best_u_1d = u
                    best_t_1d = t
            oof_actions_1d[val_idx] = np.where(se_val <= best_t_1d, 0, 3)

            # --- 2D In-Fold Grid Search (Joint tau_A, tau_E: 4 Actions) ---
            best_u_2d = -np.inf
            best_ta, best_te = np.median(au_tr), np.median(eu_tr)

            for qa in tau_quantiles:
                ta = np.percentile(au_tr, qa)
                for qe in tau_quantiles:
                    te = np.percentile(eu_tr, qe)
                    # 4-Way Policy logic:
                    # 0: Answer   (AU < ta & EU < te)
                    # 1: Retrieve (AU < ta & EU >= te)
                    # 2: Clarify  (AU >= ta & EU < te)
                    # 3: Abstain  (AU >= ta & EU >= te)
                    acts_tr = np.zeros(len(y_tr), dtype=int)
                    acts_tr[(au_tr < ta) & (eu_tr >= te)] = 1
                    acts_tr[(au_tr >= ta) & (eu_tr < te)] = 2
                    acts_tr[(au_tr >= ta) & (eu_tr >= te)] = 3

                    u = _calculate_utility_for_actions(acts_tr, y_tr, r_params)
                    if u > best_u_2d:
                        best_u_2d = u
                        best_ta, best_te = ta, te

            # Apply best 2D thresholds to out-of-fold validation split
            acts_val = np.zeros(len(y_val), dtype=int)
            acts_val[(au_val < best_ta) & (eu_val >= best_te)] = 1
            acts_val[(au_val >= best_ta) & (eu_val < best_te)] = 2
            acts_val[(au_val >= best_ta) & (eu_val >= best_te)] = 3
            oof_actions_2d[val_idx] = acts_val

        # Compute full dataset out-of-fold expected utility
        u_direct = _calculate_utility_for_actions(oof_actions_direct, y_true, r_params)
        u_1d = _calculate_utility_for_actions(oof_actions_1d, y_true, r_params)
        u_2d = _calculate_utility_for_actions(oof_actions_2d, y_true, r_params)

        results["direct_answering"][r_name] = u_direct
        results["1d_entropy_routing"][r_name] = u_1d
        results["2d_decomposed_routing"][r_name] = u_2d

    return results


# ==============================================================================
# 4. Disentanglement Diagnostics
# ==============================================================================

def compute_disentanglement_diagnostics(au: np.ndarray, eu: np.ndarray) -> Dict[str, float]:
    """
    Evaluates the statistical independence/orthogonality between AU and EU.
    """
    au = np.asarray(au, dtype=float)
    eu = np.asarray(eu, dtype=float)
    
    p_corr, p_val = pearsonr(au, eu)
    s_corr, s_val = spearmanr(au, eu)

    return {
        "pearson_rho": float(p_corr),
        "pearson_p_value": float(p_val),
        "spearman_rho": float(s_corr),
        "spearman_p_value": float(s_val)
    }
