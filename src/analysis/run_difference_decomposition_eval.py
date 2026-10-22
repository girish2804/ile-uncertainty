"""
Consolidated Difference Multi-Baseline Uncertainty & 2D Decision Routing Evaluation Runner.
Evaluates at BLEURT >= 0.0.

Uncertainty Decomposition Formulation:
- Total Uncertainty (TU): H_final (Single-Layer Final Token Entropy)
- Aleatoric Uncertainty (AU): H_mix (Multi-Layer Premature Ensembled Mixture Entropy, Ka=2)
- Epistemic Uncertainty (EU): Delta H = H_final - H_mix (Difference Epistemic Uncertainty)
- Additive Identity: AU + EU = H_mix + (H_final - H_mix) = H_final = TU

Usage:
    python run_difference_decomposition_eval.py
"""

import os
import glob
import pickle as pkl
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

GLOBAL_CORR_THRESHOLD = 0.1

CANDIDATES = [
    '../experiments/result/entropies_unified',
    'src/experiments/result/entropies_unified',
    './result/entropies_unified',
    '../../result/entropies_unified',
    'result/entropies_unified',
    'entropies_unified',
    'src/experiments',
    '../experiments',
    './src/experiments',
    'src',
    './src',
    '../src',
    '../../src',
    './result/entropies3',
    './result/entropies2',
    './results/entropies3',
    './results/entropies2',
    '../experiments/result/entropies3',
    '../experiments/result/entropies2',
    'src/experiments/result/entropies3',
    'src/experiments/result/entropies2',
    'entropies3',
    'entropies2',
    '.'
]

def find_primary_data_dir():
    for c in CANDIDATES:
        if os.path.exists(c) and any(f.endswith('.pkl') for f in os.listdir(c)):
            return c
    return '.'

def extract_layer_num(key, prefix):
    rem = key.replace(prefix, '').replace('top_', '')
    return int(rem) if rem.isdigit() else None

def _model_aliases(model):
    m = str(model).lower().replace('_', '-').replace('-instruct', '').replace('-chat', '').strip('-')
    aliases = [m, m.replace('-', '_'), m.replace('-', '')]
    if 'meta-' in m:
        aliases.extend([m.replace('meta-', ''), m.replace('meta-', '').replace('-', '')])
    else:
        aliases.append(f"meta-{m}")

    if 'qwen-1.5b' in m or 'qwen2.5-1.5b' in m:
        aliases.extend(['qwen-1.5b', 'qwen2.5-1.5b', 'qwen_1.5b', 'qwen2.5_1.5b', 'qwen2.5-1.5b-instruct', 'qwen1.5b', 'qwen2.5-1.5'])
    if 'qwen-3b' in m or 'qwen2.5-3b' in m:
        aliases.extend(['qwen-3b', 'qwen2.5-3b', 'qwen_3b', 'qwen2.5_3b', 'qwen2.5-3b-instruct', 'qwen3b', 'qwen2.5-3'])
    if 'qwen-7b' in m or 'qwen2.5-7b' in m:
        aliases.extend(['qwen-7b', 'qwen2.5-7b', 'qwen_7b', 'qwen2.5_7b', 'qwen2.5-7b-instruct', 'qwen7b', 'qwen2.5-7'])
    if 'qwen-14b' in m or 'qwen2.5-14b' in m:
        aliases.extend(['qwen-14b', 'qwen2.5-14b', 'qwen_14b', 'qwen2.5_14b', 'qwen2.5-14b-instruct', 'qwen14b', 'qwen2.5-14'])
    if 'qwen1.5-moe' in m or 'qwen-moe' in m or 'moe' in m:
        aliases.extend(['qwen1.5-moe-a2.7b', 'qwen1.5_moe_a2.7b', 'qwen1.5-moe', 'qwen_moe', 'qwen1.5-moe-chat', 'moe-a2.7b'])
    if 'llama-3.2-3b' in m or 'llama3.2-3b' in m:
        aliases.extend(['llama-3.2-3b', 'llama3.2-3b', 'llama-3-3b', 'llama-3.2-3b-instruct', 'llama3.2_3b', 'llama-3.2', 'llama3.2'])
    if 'llama-3-8b' in m or 'meta-llama-3-8b' in m or 'llama3-8b' in m or 'llama-8b' in m or 'llama8b' in m:
        aliases.extend(['llama-3-8b', 'llama3-8b', 'meta-llama-3-8b', 'meta-llama-3-8b-instruct', 'llama-3-8b-instruct', 'llama-8b', 'llama8b', 'meta-llama-3.1-8b', 'llama-3.1-8b', 'llama3.1-8b', 'meta-llama-3', 'llama3'])
    if 'phi-3-mini' in m or 'phi-3-mini-4k' in m or 'phi-3' in m:
        aliases.extend(['phi-3-mini-4k', 'phi-3-mini', 'phi-3', 'phi-3-mini-4k-instruct', 'phi3'])
    if 'phi-3-medium' in m or 'phi-3-medium-4k' in m:
        aliases.extend(['phi-3-medium-4k', 'phi-3-medium', 'phi-3-medium-4k-instruct', 'phi3-medium'])
    if '70b' in m or 'llama-3.1-70b' in m or 'llama-70b' in m:
        aliases.extend(['llama-3.1-70b', 'llama-70b', 'llama3.1-70b', 'llama-3.1-70b-instruct', 'llama3.1_70b', 'llama70b', 'llama-3.1-70b_4bit', 'llama-3.1-70b-4bit'])
    if '72b' in m or 'qwen2.5-72b' in m or 'qwen-72b' in m:
        aliases.extend(['qwen2.5-72b', 'qwen-72b', 'qwen2.5_72b', 'qwen2.5-72b-instruct', 'qwen72b', 'qwen2.5-72b_4bit', 'qwen2.5-72b-4bit'])
    if 'deepseek' in m:
        aliases.extend(['deepseek-moe-16b', 'deepseek_moe_16b', 'deepseek-moe-16b-chat', 'deepseek-moe', 'deepseek'])
    return list(set(aliases))

def _dataset_aliases(dataset):
    d = str(dataset).lower().replace('_', '-')
    aliases = [d, d.replace('-', '_')]
    if any(t in d for t in ['truth', 'truthful', 'truthfulqa']):
        aliases.extend(['truth', 'truthful', 'truthfulqa', 'truthful_qa'])
    elif any(s in d for s in ['science', 'scienceqa', 'sciq']):
        aliases.extend(['science', 'scienceqa', 'science_qa', 'sciq'])
    elif any(o in d for o in ['open', 'openbook', 'openbookqa']):
        aliases.extend(['open', 'openbook', 'openbookqa', 'openbook_qa'])
    elif any(f in d for f in ['fresh', 'freshqa']):
        aliases.extend(['fresh', 'freshqa', 'fresh_qa'])
    elif any(a in d for a in ['asqa', 'ambig']):
        aliases.extend(['asqa', 'ambig', 'asqa_long'])
    return list(set(aliases))

def calculate_auroc_and_prr(df, uncertainty_col, threshold=GLOBAL_CORR_THRESHOLD):
    if uncertainty_col not in df.columns:
        return 0.500, 0.0
    valid_mask = df[uncertainty_col].notna() & df['Correctness'].notna()
    df_clean = df[valid_mask].copy()
    if len(df_clean) == 0:
        return 0.500, 0.0
        
    correct = (df_clean['Correctness'] >= threshold).astype(int).values
    u_val = df_clean[uncertainty_col].values
    N = len(correct)
    a0 = correct.mean()
    
    if len(np.unique(correct)) > 1:
        auroc = float(roc_auc_score(1 - correct, u_val))
    else:
        auroc = 0.500

    # 1D Rejection Curve
    sort_idx = np.argsort(u_val)
    sorted_correct = correct[sort_idx]
    coverages = np.linspace(0.01, 1.0, 100)
    accuracies_method = [sorted_correct[:max(1, int(N * cov))].mean() for cov in coverages]
    accuracies_oracle = [min(max(1, int(N * cov)), correct.sum()) / max(1, int(N * cov)) for cov in coverages]
    
    area_method = np.trapz(accuracies_method, coverages)
    area_oracle = np.trapz(accuracies_oracle, coverages)
    denom = area_oracle - a0
    prr = float((area_method - a0) / denom) if denom > 1e-10 else 0.0
    
    return auroc, prr

def compute_2d_pareto_prr(au, eu, y_true, n_grid=40, n_bins=100):
    valid_mask = ~np.isnan(au) & ~np.isnan(eu) & ~np.isnan(y_true)
    au_c, eu_c, y_c = au[valid_mask], eu[valid_mask], y_true[valid_mask]
    if len(y_c) == 0 or len(np.unique(y_c)) <= 1:
        return 0.0
        
    n = len(y_c)
    base_acc = float(np.mean(y_c))
    tau_a_candidates = np.percentile(au_c, np.linspace(5, 100, n_grid))
    tau_e_candidates = np.percentile(eu_c, np.linspace(5, 100, n_grid))

    cov_list, acc_list = [], []
    for ta in tau_a_candidates:
        mask_a = au_c <= ta
        for te in tau_e_candidates:
            retained = mask_a & (eu_c <= te)
            k = np.sum(retained)
            if k == 0: continue
            cov_list.append(k / n)
            acc_list.append(np.mean(y_c[retained]))

    cov_list.append(1.0); acc_list.append(base_acc)
    cov_arr, acc_arr = np.array(cov_list), np.array(acc_list)

    grid_cov = np.linspace(0.05, 1.0, n_bins)
    pareto_acc = np.zeros_like(grid_cov)
    for i, c in enumerate(grid_cov):
        tol = 1.0 / n_bins
        nearby = (cov_arr >= c - tol) & (cov_arr <= c + tol)
        pareto_acc[i] = np.max(acc_arr[nearby]) if np.any(nearby) else acc_arr[np.argmin(np.abs(cov_arr - c))]

    for i in range(len(pareto_acc) - 2, -1, -1):
        if pareto_acc[i] < pareto_acc[i + 1]:
            pareto_acc[i] = pareto_acc[i + 1]

    aurc_2d = float(np.trapz(pareto_acc, grid_cov))
    oracle_y = np.sort(y_c)[::-1]
    oracle_acc = np.cumsum(oracle_y) / np.arange(1, n + 1)
    oracle_aurc = float(np.trapz(oracle_acc, np.arange(1, n + 1) / n))
    denom = oracle_aurc - base_acc
    return float((aurc_2d - base_acc) / denom) if denom > 1e-12 else 0.0

DEFAULT_REGIMES = {
    "Standard": {"R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3},
    "High-Safety": {"R_correct": 1.0, "C_err": -10.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3},
    "High-RAG-Cost": {"R_correct": 1.0, "C_err": -5.0, "C_rag": -1.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3},
    "Low-RAG-Success": {"R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.40, "C_abstain": -0.5, "C_clarify": -0.3},
    "Low-Safety": {"R_correct": 1.0, "C_err": -2.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -0.5, "C_clarify": -0.3},
    "Highly-Capable-RAG": {"R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.90, "C_abstain": -0.5, "C_clarify": -0.3},
    "High-Abstain-Penalty": {"R_correct": 1.0, "C_err": -5.0, "C_rag": -0.5, "P_rag_success": 0.70, "C_abstain": -4.0, "C_clarify": -0.3}
}

def _calc_util(actions, y_true, r):
    r_corr = r.get("R_correct", 1.0)
    c_err = r.get("C_err", -5.0)
    c_rag = r.get("C_rag", -0.5)
    p_rag = r.get("P_rag_success", 0.70)
    c_abs = r.get("C_abstain", -0.5)
    c_cla = r.get("C_clarify", -0.3)
    u = np.zeros_like(y_true, dtype=float)
    u[actions == 0] = np.where(y_true[actions == 0] == 1, r_corr, c_err)
    u[actions == 1] = p_rag * r_corr + (1 - p_rag) * c_err + c_rag
    u[actions == 2] = 0.85 * np.where(y_true[actions == 2] == 1, r_corr, c_err) + c_cla
    u[actions == 3] = c_abs
    return float(np.mean(u))

def run_5fold_routing(au, eu_diff, eu_gated, se_arr, ice_tu, tok_tu, y_true):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    t_quants = np.linspace(10, 90, 15)
    res = {}

    for r_name, r_params in DEFAULT_REGIMES.items():
        oof_dir = np.zeros(len(y_true), dtype=int)
        oof_1d = np.zeros(len(y_true), dtype=int)
        oof_se = np.zeros(len(y_true), dtype=int)
        oof_ice = np.zeros(len(y_true), dtype=int)
        oof_tok = np.zeros(len(y_true), dtype=int)
        oof_2d_diff = np.zeros(len(y_true), dtype=int)
        oof_2d_gated = np.zeros(len(y_true), dtype=int)

        for tr, val in kf.split(y_true):
            y_tr, y_val = y_true[tr], y_true[val]
            au_tr, au_val = au[tr], au[val]
            eud_tr, eud_val = eu_diff[tr], eu_diff[val]
            eug_tr, eug_val = eu_gated[tr], eu_gated[val]
            se_tr, se_val = se_arr[tr], se_arr[val]
            ice_tr, ice_val = ice_tu[tr], ice_tu[val]
            tok_tr, tok_val = tok_tu[tr], tok_tu[val]

            # 1. 1D Single Entropy (H_final)
            b_u1, b_t1 = -np.inf, np.median(au_tr)
            for q in t_quants:
                t = np.percentile(au_tr, q)
                u = _calc_util(np.where(au_tr <= t, 0, 3), y_tr, r_params)
                if u > b_u1: b_u1, b_t1 = u, t
            oof_1d[val] = np.where(au_val <= b_t1, 0, 3)

            # 2. Semantic Entropy (SE) Routing
            if not np.isnan(se_tr).all():
                b_use, b_tse = -np.inf, np.nanmedian(se_tr)
                for q in t_quants:
                    t = np.nanpercentile(se_tr, q)
                    u = _calc_util(np.where(se_tr <= t, 0, 3), y_tr, r_params)
                    if u > b_use: b_use, b_tse = u, t
                oof_se[val] = np.where(se_val <= b_tse, 0, 3)

            # 3. ICE Routing
            if not np.isnan(ice_tr).all():
                b_uice, b_tice = -np.inf, np.nanmedian(ice_tr)
                for q in t_quants:
                    t = np.nanpercentile(ice_tr, q)
                    u = _calc_util(np.where(ice_tr <= t, 0, 3), y_tr, r_params)
                    if u > b_uice: b_uice, b_tice = u, t
                oof_ice[val] = np.where(ice_val <= b_tice, 0, 3)

            # 4. TokUR Routing
            if not np.isnan(tok_tr).all():
                b_utok, b_ttok = -np.inf, np.nanmedian(tok_tr)
                for q in t_quants:
                    t = np.nanpercentile(tok_tr, q)
                    u = _calc_util(np.where(tok_tr <= t, 0, 3), y_tr, r_params)
                    if u > b_utok: b_utok, b_ttok = u, t
                oof_tok[val] = np.where(tok_val <= b_ttok, 0, 3)

            # 5. 2D Decomposed (Ours Difference: AU = H_mix, EU = H_final - H_mix)
            b_u2d, b_tad, b_ted = -np.inf, np.median(au_tr), np.median(eud_tr)
            for qa in t_quants:
                ta = np.percentile(au_tr, qa)
                for qe in t_quants:
                    te = np.percentile(eud_tr, qe)
                    acts = np.zeros(len(y_tr), dtype=int)
                    acts[(au_tr < ta) & (eud_tr >= te)] = 1
                    acts[(au_tr >= ta) & (eud_tr < te)] = 2
                    acts[(au_tr >= ta) & (eud_tr >= te)] = 3
                    u = _calc_util(acts, y_tr, r_params)
                    if u > b_u2d: b_u2d, b_tad, b_ted = u, ta, te

            acts_vd = np.zeros(len(y_val), dtype=int)
            acts_vd[(au_val < b_tad) & (eud_val >= b_ted)] = 1
            acts_vd[(au_val >= b_tad) & (eud_val < b_ted)] = 2
            acts_vd[(au_val >= b_tad) & (eud_val >= b_ted)] = 3
            oof_2d_diff[val] = acts_vd

            # 6. 2D Decomposed (Ours Gated MI)
            b_u2g, b_tag, b_teg = -np.inf, np.median(au_tr), np.median(eug_tr)
            for qa in t_quants:
                ta = np.percentile(au_tr, qa)
                for qe in t_quants:
                    te = np.percentile(eug_tr, qe)
                    acts = np.zeros(len(y_tr), dtype=int)
                    acts[(au_tr < ta) & (eug_tr >= te)] = 1
                    acts[(au_tr >= ta) & (eug_tr < te)] = 2
                    acts[(au_tr >= ta) & (eug_tr >= te)] = 3
                    u = _calc_util(acts, y_tr, r_params)
                    if u > b_u2g: b_u2g, b_tag, b_teg = u, ta, te

            acts_vg = np.zeros(len(y_val), dtype=int)
            acts_vg[(au_val < b_tag) & (eug_val >= b_teg)] = 1
            acts_vg[(au_val >= b_tag) & (eug_val < b_teg)] = 2
            acts_vg[(au_val >= b_tag) & (eug_val >= b_teg)] = 3
            oof_2d_gated[val] = acts_vg

        res[r_name] = {
            "Direct": _calc_util(oof_dir, y_true, r_params),
            "1D_Ent": _calc_util(oof_1d, y_true, r_params),
            "SE": _calc_util(oof_se, y_true, r_params) if not np.isnan(se_arr).all() else np.nan,
            "ICE": _calc_util(oof_ice, y_true, r_params) if not np.isnan(ice_tu).all() else np.nan,
            "TokUR": _calc_util(oof_tok, y_true, r_params) if not np.isnan(tok_tu).all() else np.nan,
            "2D_Ours_Diff": _calc_util(oof_2d_diff, y_true, r_params),
            "2D_Ours_Gated": _calc_util(oof_2d_gated, y_true, r_params)
        }
    return res

def load_run_dataframe(run, data_dir):
    with open(run['entropy_file'], 'rb') as f: raw_entropies = pkl.load(f)
    with open(run['label_file'], 'rb') as f: raw_labels = pkl.load(f)

    num_instances = len(raw_labels)
    if len(raw_entropies) > 0 and len(raw_entropies) < num_instances:
        num_instances = len(raw_entropies)

    labels = []
    for l in raw_labels[:num_instances]:
        if isinstance(l, (list, tuple, np.ndarray)): labels.append(float(l[0]) if len(l) > 0 else 0.0)
        elif isinstance(l, dict): labels.append(float(l.get('bleurt', l.get('bleurt_score', 0.0))))
        else: labels.append(float(l))

    if len(raw_entropies) >= int(2.5 * num_instances):
        chunk_stride = len(raw_entropies) // 3
        ours_entropies = raw_entropies[0 : num_instances]
        sdlg_entropies = raw_entropies[chunk_stride : chunk_stride + num_instances]
        kuhn_entropies = raw_entropies[2 * chunk_stride : 2 * chunk_stride + num_instances]
    elif len(raw_entropies) >= int(1.8 * num_instances):
        chunk_stride = len(raw_entropies) // 2
        ours_entropies = raw_entropies[0 : num_instances]
        sdlg_entropies = raw_entropies[chunk_stride : chunk_stride + num_instances]
        kuhn_entropies = None
    else:
        ours_entropies = raw_entropies[:num_instances]
        sdlg_entropies = None
        kuhn_entropies = None

    keys = list(ours_entropies[0].keys()) if ours_entropies and isinstance(ours_entropies[0], dict) else []

    # Map parameter size class optimal Ka*, Ke*
    m_clean = str(run['model']).lower().replace('_', '-')
    if any(s in m_clean for s in ['70b', '72b']):
        model_class, ka_depth, ke_depth = '70B+ (70B-72B)', 2, 60
    elif any(s in m_clean for s in ['14b', 'phi-3-medium']):
        model_class, ka_depth, ke_depth = 'Large (14B+)', 3, 32
    elif any(s in m_clean for s in ['7b', '8b']):
        model_class, ka_depth, ke_depth = 'Mid (7B-8B)', 12, 28
    else:
        model_class, ka_depth, ke_depth = 'Small (1.5B-3B)', 2, 27

    au_key = 'entropy_final' if 'entropy_final' in keys else 'entropy_mixture_top_1'
    ka_keys = [k for k in keys if k.startswith('entropy_mixture_')]
    best_ka_key = sorted(ka_keys, key=lambda x: abs((extract_layer_num(x, 'entropy_mixture_') or 0) - ka_depth))[0] if ka_keys else au_key
    best_ke_mixture_key = sorted(ka_keys, key=lambda x: abs((extract_layer_num(x, 'entropy_mixture_') or 0) - ke_depth))[0] if ka_keys else au_key

    ke_mi_keys = [k for k in keys if k.startswith('mutual_info_') and 'gated' not in k]
    best_ke_mi_key = sorted(ke_mi_keys, key=lambda x: abs((extract_layer_num(x, 'mutual_info_') or 0) - 16))[0] if ke_mi_keys else 'mutual_info'

    gated_mi_candidates = ['gated_mutual_info_top_16', 'gated_mutual_info_top_8', 'gated_mutual_info', 'gated_mutual_info_depth_75']
    gated_mi_key = next((k for k in gated_mi_candidates if any(isinstance(d, dict) and k in d for d in ours_entropies)), 'gated_mutual_info')

    mi_candidates = ['mutual_info_top_16', 'mutual_info_16', 'mutual_info_depth_16', 'mutual_info_15', 'mutual_info_top_8', 'mutual_info']
    mi_key = next((k for k in mi_candidates if any(isinstance(d, dict) and k in d for d in ours_entropies)), best_ke_mi_key)

    has_se = any(isinstance(d, dict) and 'normalised_semantic_entropy' in d for d in ours_entropies)

    if sdlg_entropies is None or kuhn_entropies is None or not has_se:
        m_aliases = _model_aliases(run['model'])
        candidate_e2_dirs = [os.path.join(data_dir, '..', 'entropies2'), './result/entropies2', 'src/experiments/result/entropies2', 'entropies2']
        for e2_dir in candidate_e2_dirs:
            if not os.path.exists(e2_dir): continue
            for fname in os.listdir(e2_dir):
                fl = fname.lower()
                if '_entropies_' in fl and any(ma in fl for ma in m_aliases):
                    try:
                        with open(os.path.join(e2_dir, fname), 'rb') as fp: e2_raw = pkl.load(fp)
                        if isinstance(e2_raw, list) and len(e2_raw) >= int(2.5 * num_instances):
                            e2_stride = len(e2_raw) // 3
                            if sdlg_entropies is None: sdlg_entropies = e2_raw[e2_stride : e2_stride + num_instances]
                            if kuhn_entropies is None: kuhn_entropies = e2_raw[2 * e2_stride : 2 * e2_stride + num_instances]
                            if not has_se:
                                ours_se_c = e2_raw[0 : num_instances]
                                for idx_i in range(min(len(ours_entropies), len(ours_se_c))):
                                    if isinstance(ours_entropies[idx_i], dict) and isinstance(ours_se_c[idx_i], dict) and 'normalised_semantic_entropy' in ours_se_c[idx_i]:
                                        ours_entropies[idx_i]['normalised_semantic_entropy'] = ours_se_c[idx_i]['normalised_semantic_entropy']
                                has_se = True
                            break
                    except Exception: pass
            if sdlg_entropies is not None and kuhn_entropies is not None and has_se: break

    h_final_vals = np.array([d.get(au_key, 0.0) for d in ours_entropies])
    h_mix_ka_vals = np.array([d.get(best_ka_key, d.get(au_key, 0.0)) for d in ours_entropies])
    h_mix_ke_vals = np.array([d.get(best_ke_mixture_key, d.get(au_key, 0.0)) for d in ours_entropies])
    epistemic_diff_vals = h_final_vals - h_mix_ke_vals

    df = pd.DataFrame({
        'Correctness': labels[:num_instances],
        'Is_Correct': (pd.Series(labels[:num_instances]) >= GLOBAL_CORR_THRESHOLD).astype(int),
        'H_final': h_final_vals,
        'H_mix_Ka': h_mix_ka_vals,
        'Epistemic_Diff': epistemic_diff_vals,
        'Gated_MI': [d.get(gated_mi_key, d.get('gated_mutual_info', 0.0)) for d in ours_entropies],
        'JSD': [d.get('jsd', 0.0) for d in ours_entropies],
        'Epistemic_MI_Ke16': [d.get(mi_key, d.get(best_ke_mi_key, d.get('mutual_info', 0.0))) for d in ours_entropies],
        'Our_Weighted_SE': [d.get('normalised_semantic_entropy', 0.0) for d in ours_entropies],
        'SDLG_SE': [d.get('normalised_semantic_entropy', 0.0) for d in sdlg_entropies] if sdlg_entropies else [d.get('normalised_semantic_entropy', 0.0) for d in ours_entropies],
        'Kuhn_SE': [d.get('normalised_semantic_entropy', np.nan) for d in kuhn_entropies] if kuhn_entropies else np.nan
    })
    
    # Total Uncertainty Combinations
    df['Ours_TU_Diff'] = df['H_mix_Ka'] + df['Epistemic_Diff']
    df['Ours_TU_Gated'] = df['H_mix_Ka'] + df['Gated_MI']
    df['Ours_TU_JSD'] = df['H_mix_Ka'] + df['JSD']
    df['Ours_TU_MI16'] = df['H_mix_Ka'] + df['Epistemic_MI_Ke16']

    # Link TokUR and ICE baselines
    m_aliases = _model_aliases(run['model'])
    d_aliases = _dataset_aliases(run['dataset'])
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(data_dir)))
    
    # TokUR
    tokur_dirs = [os.path.join(exp_dir, 'tokur', 'result'), os.path.join(exp_dir, 'tokur'), os.path.join(data_dir, '..', 'tokur', 'result')]
    for t_dir in tokur_dirs:
        if not os.path.exists(t_dir): continue
        for folder in os.listdir(t_dir):
            fl = folder.lower()
            if any(ma in fl for ma in m_aliases) and any(da in fl for da in d_aliases):
                ef = os.path.join(t_dir, folder, 'entropies_tokur.pkl')
                if os.path.exists(ef):
                    try:
                        with open(ef, 'rb') as tf: tok_ent = pkl.load(tf)
                        def _ext_tok(e, k1, k2):
                            if not isinstance(e, dict): return np.nan
                            sub = e.get('entropy_dict', e) if isinstance(e.get('entropy_dict'), dict) else e
                            return sub.get(k1, sub.get(k2, np.nan))
                        tok_au = np.array([_ext_tok(e, 'tokur_au', 'entropy_final') for e in tok_ent[:num_instances]], dtype=float)
                        tok_mi = np.array([_ext_tok(e, 'tokur_mi', 'mutual_info') for e in tok_ent[:num_instances]], dtype=float)
                        if np.isfinite(tok_au).any() or np.isfinite(tok_mi).any():
                            df['TokUR_AU'] = tok_au
                            df['TokUR_EU'] = tok_mi
                            df['TokUR_Total'] = np.where(np.isfinite(tok_au) & np.isfinite(tok_mi), tok_au + tok_mi, np.nan)
                            break
                    except Exception: pass

    # ICE
    ice_dirs = [os.path.join(exp_dir, 'ice', 'result'), os.path.join(exp_dir, 'ice'), os.path.join(data_dir, '..', 'ice', 'result')]
    for i_dir in ice_dirs:
        if not os.path.exists(i_dir): continue
        for fname in os.listdir(i_dir):
            fl = fname.lower()
            if (fl.endswith('_summary.pkl') or fl.endswith('_summary.csv')) and any(ma in fl for ma in m_aliases) and any(da in fl for da in d_aliases):
                try:
                    fpath = os.path.join(i_dir, fname)
                    ice_df = pd.read_csv(fpath) if fpath.endswith('.csv') else pd.DataFrame(pkl.load(open(fpath, 'rb')))
                    df['ICE_AU'] = ice_df['aleatoric_uncertainty'].values[:num_instances]
                    df['ICE_EU'] = ice_df['epistemic_uncertainty'].values[:num_instances]
                    df['ICE_Total'] = ice_df['total_uncertainty'].values[:num_instances]
                    break
                except Exception: pass

    return df

def main():
    valid_dirs = []
    for c in CANDIDATES:
        if os.path.exists(c) and any(f.endswith('.pkl') for f in os.listdir(c)):
            abs_c = os.path.abspath(c)
            if abs_c not in valid_dirs:
                valid_dirs.append(abs_c)

    data_dir = valid_dirs[0] if valid_dirs else '.'
    print(f"[*] Primary Data Directory: {os.path.abspath(data_dir)}")
    print(f"[*] Valid Candidate Directories: {valid_dirs}")
    print(f"[*] Evaluation Threshold: BLEURT >= {GLOBAL_CORR_THRESHOLD}\n")

    runs = []
    seen_models = set()
    for d_path in valid_dirs:
        raw_files = glob.glob(os.path.join(d_path, "*.pkl"))
        entropy_files = sorted(list(set([f for f in raw_files if "_entropies_" in os.path.basename(f) or os.path.basename(f).startswith("sdlg_entropies_")])))

        for f in entropy_files:
            base = os.path.basename(f)
            label_file = f.replace("_entropies_", "_label_").replace("sdlg_entropies_", "sdlg_label_")
            if not os.path.exists(label_file):
                if 'qwen-1.5b' in label_file: label_file = label_file.replace('qwen-1.5b', 'qwen2.5-1.5b')
                elif 'qwen2.5-1.5b' in label_file: label_file = label_file.replace('qwen2.5-1.5b', 'qwen-1.5b')
                
            if not os.path.exists(label_file): continue
            
            parts = base.replace(".pkl", "").split("_")
            model_name = "unknown"
            dataset_name = "truthfulqa"
            for p in parts:
                if any(m in p.lower() for m in ['llama', 'qwen', 'phi', 'opt', 'mistral', 'gemma']): model_name = p
                if any(d in p.lower() for d in ['truth', 'sciq', 'openbook', 'science', 'ambig', 'asqa', 'fresh', 'freshqa']):
                    dataset_name = 'scienceqa' if 'science' in p.lower() or 'sciq' in p.lower() else ('openbookqa' if 'open' in p.lower() else ('freshqa' if 'fresh' in p.lower() else ('asqa' if ('asqa' in p.lower() or 'ambig' in p.lower()) else ('truthfulqa' if 'truth' in p.lower() else p.lower()))))

            run_key = (dataset_name.lower(), model_name.lower())
            if run_key in seen_models: continue
            seen_models.add(run_key)
            
            runs.append({
                'name': f"{model_name} on {dataset_name}",
                'model': model_name,
                'dataset': dataset_name,
                'entropy_file': f,
                'label_file': label_file
            })

    print(f"[*] Discovered {len(runs)} unique evaluation runs across datasets.\n")

    # =========================================================================
    # TABLE 1: TOTAL UNCERTAINTY COMPARISON (AUROC & PRR)
    # =========================================================================
    tu_baselines = [
        ('H_final (Single-Layer)', 'H_final'),
        ('Ours TU [Difference]', 'Ours_TU_Diff'),
        ('Ours TU [Gated MI]', 'Ours_TU_Gated'),
        ('Ours TU [JSD]', 'Ours_TU_JSD'),
        ('Ours TU [MI_16]', 'Ours_TU_MI16'),
        ('Kuhn SE', 'Kuhn_SE'),
        ('SDLG SE', 'SDLG_SE'),
        ('ICE Total', 'ICE_Total'),
        ('TokUR Total', 'TokUR_Total'),
    ]

    t1_auroc_rows, t1_prr_rows = [], []
    t2a_prr_rows = []
    t2b_rho_rows = []
    t2c_decomp_auroc_rows = []
    all_routing_results = {}

    for r in runs:
        df = load_run_dataframe(r, data_dir)
        acc_str = f"{df['Is_Correct'].mean() * 100:.1f}%"
        y_true = df['Is_Correct'].values
        
        # --- 1. Total Uncertainty Metrics ---
        row_a = [r['name'], acc_str]
        row_p = [r['name'], acc_str]
        for lbl, col in tu_baselines:
            if col in df.columns and df[col].notna().any():
                auroc, prr = calculate_auroc_and_prr(df, col)
                row_a.append(f"{auroc:.4f}")
                row_p.append(f"{prr:.4f}")
            else:
                row_a.append("-")
                row_p.append("-")
        t1_auroc_rows.append(row_a)
        t1_prr_rows.append(row_p)

        # --- 2. Decomposed Components ---
        au_mix = df['H_mix_Ka'].values
        tu_final = df['H_final'].values
        eu_diff = df['Epistemic_Diff'].values
        eu_gated = df['Gated_MI'].values if 'Gated_MI' in df.columns else df['Epistemic_MI_Ke16'].values
        eu_jsd = df['JSD'].values if 'JSD' in df.columns else df['Epistemic_MI_Ke16'].values
        eu_mi16 = df['Epistemic_MI_Ke16'].values

        # 2A: 2D Pareto PRR
        prr2d_ours_diff = compute_2d_pareto_prr(au_mix, eu_diff, y_true)
        prr2d_ours_gated = compute_2d_pareto_prr(au_mix, eu_gated, y_true)
        prr2d_ours_jsd = compute_2d_pareto_prr(au_mix, eu_jsd, y_true)
        prr2d_ours_mi16 = compute_2d_pareto_prr(au_mix, eu_mi16, y_true)

        has_ice = 'ICE_AU' in df.columns and df['ICE_AU'].notna().any() and 'ICE_EU' in df.columns and df['ICE_EU'].notna().any()
        if has_ice:
            au_ice = df['ICE_AU'].values.astype(float)
            eu_ice = df['ICE_EU'].values.astype(float)
            prr2d_ice = f"{compute_2d_pareto_prr(au_ice, eu_ice, y_true):.4f}"
            rho_ice = f"{spearmanr(au_ice, eu_ice)[0]:.4f}"
            roc_ice_au = f"{calculate_auroc_and_prr(df, 'ICE_AU')[0]:.4f}"
            roc_ice_eu = f"{calculate_auroc_and_prr(df, 'ICE_EU')[0]:.4f}"
        else:
            prr2d_ice, rho_ice, roc_ice_au, roc_ice_eu = "-", "-", "-", "-"

        has_tok = 'TokUR_AU' in df.columns and df['TokUR_AU'].notna().any() and 'TokUR_EU' in df.columns and df['TokUR_EU'].notna().any()
        if has_tok:
            au_tok = df['TokUR_AU'].values.astype(float)
            eu_tok = df['TokUR_EU'].values.astype(float)
            prr2d_tok = f"{compute_2d_pareto_prr(au_tok, eu_tok, y_true):.4f}"
            rho_tok = f"{spearmanr(au_tok, eu_tok)[0]:.4f}"
            roc_tok_au = f"{calculate_auroc_and_prr(df, 'TokUR_AU')[0]:.4f}"
            roc_tok_eu = f"{calculate_auroc_and_prr(df, 'TokUR_EU')[0]:.4f}"
        else:
            prr2d_tok, rho_tok, roc_tok_au, roc_tok_eu = "-", "-", "-", "-"

        t2a_prr_rows.append([
            r['name'],
            f"{prr2d_ours_diff:.4f}",
            f"{prr2d_ours_gated:.4f}",
            f"{prr2d_ours_jsd:.4f}",
            f"{prr2d_ours_mi16:.4f}",
            prr2d_ice,
            prr2d_tok
        ])

        # 2B: Disentanglement Orthogonality (rho)
        rho_ours_diff = f"{spearmanr(au_mix, eu_diff)[0]:.4f}"
        rho_ours_gated = f"{spearmanr(au_mix, eu_gated)[0]:.4f}"
        rho_ours_jsd = f"{spearmanr(au_mix, eu_jsd)[0]:.4f}"
        rho_ours_mi16 = f"{spearmanr(au_mix, eu_mi16)[0]:.4f}"
        t2b_rho_rows.append([
            r['name'],
            rho_ours_diff,
            rho_ours_gated,
            rho_ours_jsd,
            rho_ours_mi16,
            rho_ice,
            rho_tok
        ])

        # 2C: Decomposed Components AUROC
        roc_ours_tu_final = f"{calculate_auroc_and_prr(df, 'H_final')[0]:.4f}"
        roc_ours_aumix = f"{calculate_auroc_and_prr(df, 'H_mix_Ka')[0]:.4f}"
        roc_ours_eudiff = f"{calculate_auroc_and_prr(df, 'Epistemic_Diff')[0]:.4f}"
        roc_ours_eugated = f"{calculate_auroc_and_prr(df, 'Gated_MI')[0]:.4f}" if 'Gated_MI' in df.columns else "-"
        roc_ours_eujsd = f"{calculate_auroc_and_prr(df, 'JSD')[0]:.4f}" if 'JSD' in df.columns else "-"
        roc_ours_eumi16 = f"{calculate_auroc_and_prr(df, 'Epistemic_MI_Ke16')[0]:.4f}"

        t2c_decomp_auroc_rows.append([
            r['name'],
            roc_ours_tu_final, roc_ours_aumix, roc_ice_au, roc_tok_au,
            roc_ours_eudiff, roc_ours_eugated, roc_ours_eujsd, roc_ours_eumi16, roc_ice_eu, roc_tok_eu
        ])

        # --- 3. 5-Fold Decision Routing ---
        se_arr = df['SDLG_SE'].values if 'SDLG_SE' in df.columns else (df['Kuhn_SE'].values if 'Kuhn_SE' in df.columns else np.full_like(y_true, np.nan, dtype=float))
        ice_tu = df['ICE_Total'].values if 'ICE_Total' in df.columns else np.full_like(y_true, np.nan, dtype=float)
        tok_tu = df['TokUR_Total'].values if 'TokUR_Total' in df.columns else np.full_like(y_true, np.nan, dtype=float)
        all_routing_results[r['name']] = run_5fold_routing(au_mix, eu_diff, eu_gated, se_arr, ice_tu, tok_tu, y_true)

    # Print Table 1A: Total Uncertainty AUROC
    h_tu = ["Model", "Acc"] + [lbl for lbl, _ in tu_baselines]
    print("=" * 130)
    print(f"TABLE 1A: TOTAL UNCERTAINTY (AUROC) - (BLEURT >= {GLOBAL_CORR_THRESHOLD})")
    print("=" * 130)
    print(tabulate(t1_auroc_rows, headers=h_tu, tablefmt="github"))
    print("\n")

    # Print Table 1B: Total Uncertainty PRR
    print("=" * 130)
    print(f"TABLE 1B: TOTAL UNCERTAINTY (PRR) - (BLEURT >= {GLOBAL_CORR_THRESHOLD})")
    print("=" * 130)
    print(tabulate(t1_prr_rows, headers=h_tu, tablefmt="github"))
    print("\n")

    # Print Table 2A: 2D Pareto Rejection Ratio
    h_2a = [
        "Model",
        "Ours 2D PRR [Diff EU]",
        "Ours 2D PRR [Gated MI]",
        "Ours 2D PRR [JSD]",
        "Ours 2D PRR [MI_16]",
        "ICE 2D PRR",
        "TokUR 2D PRR"
    ]
    print("=" * 125)
    print(f"TABLE 2A: 2D PARETO PREDICTION REJECTION RATIO (PRR_2D) - (BLEURT >= {GLOBAL_CORR_THRESHOLD})")
    print("=" * 125)
    print(tabulate(t2a_prr_rows, headers=h_2a, tablefmt="github"))
    print("\n")

    # Print Table 2B: Disentanglement Correlation
    h_2b = [
        "Model",
        "Ours rho(AU, Diff EU)",
        "Ours rho(AU, Gated MI)",
        "Ours rho(AU, JSD)",
        "Ours rho(AU, MI_16)",
        "ICE rho(AU, EU)",
        "TokUR rho(AU, EU)"
    ]
    print("=" * 125)
    print(f"TABLE 2B: UNCERTAINTY DISENTANGLEMENT CORRELATION (rho(AU, EU)) - [Lower/Near-Zero is Better]")
    print("=" * 125)
    print(tabulate(t2b_rho_rows, headers=h_2b, tablefmt="github"))
    print("\n")

    # Print Table 2C: Decomposed Components AUROC Breakdown
    h_2c = [
        "Model",
        "Total (H_final)", "Ours AU(mix)", "ICE AU", "TokUR AU",
        "Ours EU(Diff)", "Ours EU(Gated)", "Ours EU(JSD)", "Ours EU(MI_16)", "ICE EU", "TokUR EU"
    ]
    print("=" * 135)
    print(f"TABLE 2C: DECOMPOSED INDIVIDUAL COMPONENTS AUROC - (BLEURT >= {GLOBAL_CORR_THRESHOLD})")
    print("=" * 135)
    print(tabulate(t2c_decomp_auroc_rows, headers=h_2c, tablefmt="github"))
    print("\n")

    # =========================================================================
    # TABLE 3: 5-FOLD CV DECISION ROUTING UTILITY ACROSS ALL MODELS
    # =========================================================================
    print("=" * 130)
    print("TABLE 3: 5-FOLD CV EXPECTED DECISION UTILITY SUMMARY (ACROSS ALL MODELS)")
    print("=" * 130)

    for r_name in DEFAULT_REGIMES.keys():
        print(f"\n>>> Economic Cost Regime: [{r_name}]")
        regime_rows = []
        for m_name, res in all_routing_results.items():
            r_entry = res[r_name]
            u_dir = f"{r_entry['Direct']:.4f}"
            u_1d = f"{r_entry['1D_Ent']:.4f}"
            u_se = f"{r_entry['SE']:.4f}" if not np.isnan(r_entry['SE']) else "-"
            u_ice = f"{r_entry['ICE']:.4f}" if not np.isnan(r_entry['ICE']) else "-"
            u_tok = f"{r_entry['TokUR']:.4f}" if not np.isnan(r_entry['TokUR']) else "-"
            u_2d_diff = f"{r_entry['2D_Ours_Diff']:.4f}"
            u_2d_gated = f"{r_entry['2D_Ours_Gated']:.4f}"
            
            # Gain over 1D entropy routing for difference decomposition
            gain_diff = r_entry['2D_Ours_Diff'] - r_entry['1D_Ent']
            gain_str = f"+{gain_diff:.4f}" if gain_diff >= 0 else f"{gain_diff:.4f}"
            
            regime_rows.append([m_name, u_dir, u_1d, u_se, u_ice, u_tok, u_2d_diff, u_2d_gated, gain_str])
            
        h_rout = ["Model", "Direct (No UQ)", "1D Entropy", "SDLG SE", "ICE Decomp", "TokUR Decomp", "2D Ours (Diff)", "2D Ours (Gated)", "Gain (Diff - 1D)"]
        print(tabulate(regime_rows, headers=h_rout, tablefmt="github"))

    print("\n")

if __name__ == "__main__":
    main()
