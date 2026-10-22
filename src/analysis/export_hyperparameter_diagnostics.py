import os
import glob
import pickle as pkl
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr


def _find_directories():
    """Dynamically resolve root and data directories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    candidate_dirs = [
        os.path.join(script_dir, "..", "experiments", "result", "entropies_unified"),
        os.path.join(script_dir, "..", "experiments", "result", "entropies3"),
        os.path.join(script_dir, "..", "experiments", "result", "entropies2"),
        os.path.join(project_root, "result", "entropies_unified"),
        os.path.join(project_root, "result", "entropies3"),
        os.path.join(project_root, "result", "entropies2"),
        os.path.join(project_root, "src"),
        os.path.join(project_root, "entropies_unified"),
        os.path.join(project_root, "entropies3"),
        os.path.join(project_root, "entropies2"),
        os.path.join(os.getcwd(), "result", "entropies_unified"),
        os.path.join(os.getcwd(), "result", "entropies2"),
        os.path.join(os.getcwd(), "experiments", "result", "entropies2"),
        os.path.join(os.getcwd(), "src"),
    ]

    entropy_dir = None
    for d in candidate_dirs:
        if os.path.exists(d) and glob.glob(os.path.join(d, "*_entropies_*.pkl")):
            entropy_dir = d
            break

    if entropy_dir is None:
        entropy_dir = candidate_dirs[0]

    output_fig_dir = os.path.join(project_root, "figures", "hyperparameter_analysis")
    os.makedirs(output_fig_dir, exist_ok=True)

    candidate_tokur_dirs = [
        os.path.join(script_dir, "..", "experiments", "tokur", "result"),
        os.path.join(project_root, "experiments", "tokur", "result"),
        os.path.join(project_root, "results", "tokur"),
        os.path.join(project_root, "tokur", "result"),
        os.path.join(os.getcwd(), "experiments", "tokur", "result"),
        os.path.join(os.getcwd(), "tokur", "result"),
    ]

    candidate_ice_dirs = [
        os.path.join(script_dir, "..", "experiments", "ice", "result"),
        os.path.join(project_root, "experiments", "ice", "result"),
        os.path.join(project_root, "results", "ice"),
        os.path.join(project_root, "ice", "result"),
        os.path.join(os.getcwd(), "experiments", "ice", "result"),
        os.path.join(os.getcwd(), "ice", "result"),
    ]

    return entropy_dir, candidate_tokur_dirs, candidate_ice_dirs, output_fig_dir, project_root


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
    return list(set(aliases))


def load_run_full_data(entropy_file, candidate_tokur_dirs, candidate_ice_dirs):
    """Load raw entropies, labels, layerwise ensembling metrics, and baseline data."""
    label_file = entropy_file.replace("_entropies_", "_label_")
    question_file = entropy_file.replace("_entropies_", "_question_")

    if not os.path.exists(label_file):
        return None

    try:
        with open(entropy_file, 'rb') as f: raw_entropies = pkl.load(f)
        with open(label_file, 'rb') as f: labels = pkl.load(f)
        questions = pkl.load(open(question_file, 'rb')) if os.path.exists(question_file) else []
    except Exception as e:
        print(f"Error loading {entropy_file}: {e}")
        return None

    num_instances = len(labels)
    if num_instances == 0 or len(raw_entropies) == 0:
        return None

    ours_entropies = raw_entropies[:num_instances]
    sample_keys = list(ours_entropies[0].keys())

    au_key = 'entropy_final'
    if au_key not in ours_entropies[0]:
        au_candidates = [k for k in sample_keys if 'entropy_final' in k]
        if au_candidates: au_key = au_candidates[0]

    # Asymmetric layer target depths: Ka = 2 (aleatoric), Ke = 16 (epistemic)
    ka_target, ke_target = 2, 16
    
    def extract_layer_num(key, prefix):
        rem = key.replace(prefix, '').replace('top_', '')
        return int(rem) if rem.isdigit() else None
        
    em_keys = [k for k in sample_keys if k.startswith('entropy_mixture_') and extract_layer_num(k, 'entropy_mixture_') is not None]
    mi_keys = [k for k in sample_keys if k.startswith('mutual_info_') and extract_layer_num(k, 'mutual_info_') is not None]
    
    best_ka_key = 'entropy_mixture'
    if em_keys:
        em_keys.sort(key=lambda x: abs(extract_layer_num(x, 'entropy_mixture_') - ka_target))
        best_ka_key = em_keys[0]
        
    best_ke_key = 'mutual_info'
    if mi_keys:
        mi_keys.sort(key=lambda x: abs(extract_layer_num(x, 'mutual_info_') - ke_target))
        best_ke_key = mi_keys[0]

    df = pd.DataFrame({
        'Question': questions[:num_instances] if len(questions) >= num_instances else range(num_instances),
        'Correctness': labels[:num_instances],
        'Our_Weighted_SE': [d.get('normalised_semantic_entropy', 0.0) for d in ours_entropies],
        'entropy_final': [d.get(au_key, 0.0) for d in ours_entropies],
        'entropy_mixture': [d.get(best_ka_key, d.get('entropy_mixture', 0.0)) for d in ours_entropies],
        'mutual_info': [d.get(best_ke_key, d.get('mutual_info', 0.0)) for d in ours_entropies],
    })

    # Optional epistemic / disagreement metrics
    for opt_key in ['jsd', 'kl', 'cv', 'epkl', 'tvd', 'bhattacharyya']:
        if opt_key in sample_keys:
            df[opt_key] = [d.get(opt_key, 0.0) for d in ours_entropies]

    # Chunk strides for SDLG & Kuhn SE baselines
    if len(raw_entropies) >= int(2.5 * num_instances):
        chunk_stride = len(raw_entropies) // 3
        sdlg_entropies = raw_entropies[chunk_stride : chunk_stride + num_instances]
        kuhn_entropies = raw_entropies[2 * chunk_stride : 2 * chunk_stride + num_instances]
        df['SDLG_Baseline_SE'] = [d.get('normalised_semantic_entropy', 0.0) for d in sdlg_entropies]
        df['Kuhn_SE'] = [d.get('normalised_semantic_entropy', 0.0) for d in kuhn_entropies]
    elif len(raw_entropies) >= int(1.8 * num_instances):
        chunk_stride = len(raw_entropies) // 2
        sdlg_entropies = raw_entropies[chunk_stride : chunk_stride + num_instances]
        df['SDLG_Baseline_SE'] = [d.get('normalised_semantic_entropy', 0.0) for d in sdlg_entropies]
        df['Kuhn_SE'] = np.nan
    else:
        df['SDLG_Baseline_SE'] = [d.get('normalised_semantic_entropy', 0.0) for d in ours_entropies]
        df['Kuhn_SE'] = np.nan

    fname = os.path.basename(entropy_file).replace(".pkl", "")
    parts = fname.split("_entropies_")
    if len(parts) == 2:
        meta = parts[1].replace("_new", "").split("_")
        dataset_name, model_name = meta[0], "_".join(meta[1:])
    else:
        dataset_name, model_name = "unknown", fname

    # Discover all layer ensembling keys
    em_layer_dict = {}
    mi_layer_dict = {}
    for k in sample_keys:
        m_em = re.match(r"^entropy_mixture_top_(\d+)$", k) or (re.match(r"^entropy_mixture_(\d+)$", k) if k.split('_')[-1].isdigit() else None)
        if m_em:
            layer_idx = int(m_em.group(1))
            em_layer_dict[layer_idx] = [d.get(k, 0.0) for d in ours_entropies]

        m_mi = re.match(r"^mutual_info_top_(\d+)$", k) or (re.match(r"^mutual_info_(\d+)$", k) if k.split('_')[-1].isdigit() else None)
        if m_mi:
            layer_idx = int(m_mi.group(1))
            mi_layer_dict[layer_idx] = [d.get(k, 0.0) for d in ours_entropies]

    # TokUR
    m_aliases = _model_aliases(model_name)
    d_aliases = _dataset_aliases(dataset_name)

    tokur_found = False
    for tokur_dir in candidate_tokur_dirs:
        if tokur_found or not os.path.exists(tokur_dir): continue
        for folder in os.listdir(tokur_dir):
            fl = folder.lower()
            if any(ma in fl for ma in m_aliases) and any(da in fl for da in d_aliases):
                ef = os.path.join(tokur_dir, folder, "entropies_tokur.pkl")
                if os.path.exists(ef):
                    try:
                        with open(ef, "rb") as tf: tok_ent = pkl.load(tf)
                        n_t = len(tok_ent)
                        def _ext_tok(e, k1, k2):
                            if not isinstance(e, dict): return np.nan
                            sub = e.get('entropy_dict', e) if isinstance(e.get('entropy_dict'), dict) else e
                            return sub.get(k1, sub.get(k2, np.nan))
                        tok_au = [_ext_tok(e, 'tokur_au', 'entropy_final') for e in tok_ent[:num_instances]]
                        tok_mi = [_ext_tok(e, 'tokur_mi', 'mutual_info') for e in tok_ent[:num_instances]]
                        if np.isfinite([x for x in tok_au if x is not None and not np.isnan(x)]).any() or np.isfinite([x for x in tok_mi if x is not None and not np.isnan(x)]).any():
                            df['tokur_au'] = pd.array(tok_au + [None] * max(0, num_instances - n_t), dtype=object)
                            df['tokur_mi'] = pd.array(tok_mi + [None] * max(0, num_instances - n_t), dtype=object)
                            tokur_found = True
                            break
                    except Exception: pass

    # ICE
    ice_found = False
    for ice_dir in candidate_ice_dirs:
        if ice_found or not os.path.exists(ice_dir): continue
        for ice_f in os.listdir(ice_dir):
            fl = ice_f.lower()
            if (fl.endswith("_summary.pkl") or fl.endswith("_summary.csv")) and any(ma in fl for ma in m_aliases) and any(da in fl for da in d_aliases):
                fpath = os.path.join(ice_dir, ice_f)
                try:
                    ice_df = pd.read_csv(fpath) if fpath.endswith(".csv") else pd.DataFrame(pkl.load(open(fpath, "rb")))
                    n_i = len(ice_df)
                    def _pad(col): return pd.array(list(col.values[:num_instances]) + [None] * max(0, num_instances - n_i), dtype=object)
                    df['ice_epistemic'] = _pad(ice_df['epistemic_uncertainty'])
                    df['ice_aleatoric'] = _pad(ice_df['aleatoric_uncertainty'])
                    df['ice_total']     = _pad(ice_df['total_uncertainty'])
                    ice_found = True
                    break
                except Exception: pass

    return {
        'df': df,
        'model': model_name,
        'dataset': dataset_name,
        'run_name': f"{model_name.upper()} ({dataset_name})",
        'file_tag': f"{dataset_name}_{model_name}",
        'num_samples': num_instances,
        'em_layers': em_layer_dict,
        'mi_layers': mi_layer_dict
    }


def export_hyperparameter_charts():
    """Generates and exports diagnostic plots and tables for hyperparameter re-evaluation."""
    entropy_dir, candidate_tokur_dirs, candidate_ice_dirs, fig_dir, project_root = _find_directories()
    print("=" * 80)
    print("HYPERPARAMETER RE-EVALUATION & CHART EXPORTER")
    print(f"Loading data from: {entropy_dir}")
    print(f"Exporting charts to: {fig_dir}")
    print("=" * 80)

    entropy_files = sorted(glob.glob(os.path.join(entropy_dir, "*_entropies_*.pkl")))
    if not entropy_files:
        print(f"No entropy pickle files found in {entropy_dir}.")
        return

    candidate_thresholds = sorted([
        -0.70, -0.60, -0.50, -0.40, -0.35, -0.30, -0.20, -0.15, -0.10, -0.05,
        0.0, 0.03, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40
    ])

    # =========================================================================
    # VISUAL DESIGN SYSTEM: High-contrast grouping of Ours vs. Baselines
    # =========================================================================
    metrics_config = {
        # --- Group 1: Ours (Thick Solid Lines, Vibrant Emerald & Indigo) ---
        'AU (entropy_mixture)':       {'col': 'entropy_mixture',  'color': '#059669', 'ls': '-',  'marker': 'o', 'lw': 2.8, 'zorder': 10}, # Emerald Green
        'Epistemic EU (mutual_info)': {'col': 'mutual_info',      'color': '#2563eb', 'ls': '-',  'marker': '^', 'lw': 2.8, 'zorder': 10}, # Royal Blue
        'Our Weighted SE':            {'col': 'Our_Weighted_SE',  'color': '#e11d48', 'ls': '-',  'marker': 'D', 'lw': 2.4, 'zorder': 9},  # Rose / Crimson

        # --- Group 2: Semantic Entropy Baselines (Dashed Lines, Amber & Violet) ---
        'SDLG Baseline SE':           {'col': 'SDLG_Baseline_SE', 'color': '#d97706', 'ls': '--', 'marker': 's', 'lw': 2.0, 'zorder': 8},  # Amber
        'Kuhn Semantic Entropy (SE)': {'col': 'Kuhn_SE',          'color': '#7c3aed', 'ls': '--', 'marker': 'x', 'lw': 2.0, 'zorder': 8},  # Violet / Purple

        # --- Group 3: Competing Decomposed Baselines (Dash-Dot, Teal, Cyan, Rust) ---
        'TokUR Epistemic (EU)':       {'col': 'tokur_mi',         'color': '#0d9488', 'ls': '-.', 'marker': 'P', 'lw': 1.8, 'zorder': 7},  # Teal
        'TokUR Aleatoric (AU)':       {'col': 'tokur_au',         'color': '#06b6d4', 'ls': '-.', 'marker': '*', 'lw': 1.8, 'zorder': 7},  # Cyan
        'ICE Epistemic (EU)':         {'col': 'ice_epistemic',    'color': '#c2410c', 'ls': '-.', 'marker': 'h', 'lw': 1.8, 'zorder': 6},  # Rust Orange
        'ICE Total (TU)':             {'col': 'ice_total',        'color': '#be185d', 'ls': '-.', 'marker': '+', 'lw': 1.8, 'zorder': 6},  # Magenta
        'ICE Aleatoric (AU)':         {'col': 'ice_aleatoric',    'color': '#b45309', 'ls': '-.', 'marker': 'd', 'lw': 1.8, 'zorder': 6},  # Sienna

        # --- Group 4: Single-Layer Baseline (Dotted Charcoal) ---
        'AU (entropy_final)':         {'col': 'entropy_final',    'color': '#475569', 'ls': ':',  'marker': 'v', 'lw': 1.8, 'zorder': 5},  # Slate Charcoal
    }

    summary_rows = []

    for fpath in entropy_files:
        run = load_run_full_data(fpath, candidate_tokur_dirs, candidate_ice_dirs)
        if run is None: continue

        df = run['df']
        run_name = run['run_name']
        file_tag = run['file_tag']
        scores_arr = df['Correctness'].values

        valid_thresholds = [th for th in candidate_thresholds if len(np.unique((scores_arr < th).astype(int))) >= 2]
        if not valid_thresholds: continue

        print(f"\nProcessing Run: {run_name} (N={len(df)} samples)")
        print("-" * 60)

        # -------------------------------------------------------------
        # CHART 1: AUROC vs BLEURT Threshold Sweep (All Metrics)
        # -------------------------------------------------------------
        results = {label: [] for label in metrics_config.keys()}
        for th in valid_thresholds:
            y_true = (scores_arr < th).astype(int)
            for label, cfg in metrics_config.items():
                col = cfg['col']
                if col in df.columns and df[col].notna().any():
                    valid_mask = pd.notna(df[col].values) & (df[col].values != None)
                    if np.sum(valid_mask) >= 5 and len(np.unique(y_true[valid_mask])) >= 2:
                        try:
                            score = roc_auc_score(y_true[valid_mask], np.array(df[col].values[valid_mask], dtype=float))
                        except Exception:
                            score = np.nan
                    else:
                        score = np.nan
                else:
                    score = np.nan
                results[label].append(score)

        plt.figure(figsize=(10.5, 6), dpi=150)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.grid(True, linestyle='--', alpha=0.5)

        for label, cfg in metrics_config.items():
            pairs = [(th, val) for th, val in zip(valid_thresholds, results[label]) if not np.isnan(val)]
            if pairs:
                xs, ys = zip(*pairs)
                plt.plot(xs, ys, label=label, color=cfg['color'], linestyle=cfg['ls'], marker=cfg['marker'], linewidth=cfg['lw'], markersize=6, alpha=0.9, zorder=cfg.get('zorder', 5))

        plt.axvline(-0.15, color='#e74c3c', linestyle=':', label='Target Threshold (-0.15)', linewidth=1.5)
        plt.axhline(0.5, color='gray', linestyle=':', label='Random Guess (0.50)', linewidth=1.2)

        plt.title(f"AUROC Performance vs. Correctness Threshold: {run_name}", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Correctness Threshold (BLEURT Score)", fontsize=11, labelpad=8)
        plt.ylabel("AUROC Score (Higher is Better)", fontsize=11, labelpad=8)
        plt.ylim(0.35, 1.02)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9.5, frameon=True, shadow=True)
        plt.tight_layout()

        chart1_path = os.path.join(fig_dir, f"chart1_threshold_sweep_{file_tag}.png")
        plt.savefig(chart1_path, bbox_inches='tight')
        plt.close()
        print(f"  -> Exported Chart 1: {chart1_path}")

        # -------------------------------------------------------------
        # CHART 2: Layer Ensembling Depth ($K$) Sweep for Aleatoric & Epistemic
        # -------------------------------------------------------------
        em_layers = run['em_layers']
        mi_layers = run['mi_layers']
        sorted_layers = sorted(em_layers.keys())

        if sorted_layers:
            plt.figure(figsize=(11, 6), dpi=150)
            plt.grid(True, linestyle='--', alpha=0.5)

            rep_thresholds = [th for th in [-0.50, -0.35, -0.20] if th in valid_thresholds]
            if not rep_thresholds: rep_thresholds = [valid_thresholds[len(valid_thresholds)//2]]

            th_colors = {'-0.50': '#3498db', '-0.35': '#2ecc71', '-0.20': '#e67e22'}

            for th in rep_thresholds:
                y_true = (scores_arr < th).astype(int)
                col_c = th_colors.get(f"{th:.2f}", '#9b59b6')

                # Aleatoric (entropy_mixture_top_K)
                em_aurocs = [roc_auc_score(y_true, em_layers[L]) for L in sorted_layers]
                plt.plot(sorted_layers, em_aurocs, label=f'Aleatoric AU (Thresh {th})', color=col_c, linestyle='-', marker='o', linewidth=2.0)

                # Epistemic (mutual_info_top_K)
                if mi_layers:
                    mi_aurocs = [roc_auc_score(y_true, mi_layers[L]) for L in sorted_layers if L in mi_layers]
                    if len(mi_aurocs) == len(sorted_layers):
                        plt.plot(sorted_layers, mi_aurocs, label=f'Epistemic EU (Thresh {th})', color=col_c, linestyle='--', marker='^', linewidth=2.0)

                # Flat SDLG baseline
                sdlg_val = roc_auc_score(y_true, df['SDLG_Baseline_SE'].values)
                plt.axhline(sdlg_val, color=col_c, linestyle=':', alpha=0.7, label=f'SDLG Baseline (Thresh {th}: {sdlg_val:.3f})')

            plt.title(f"Ensembling Depth Sweep: AUROC vs Top Layers (K): {run_name}", fontsize=13, fontweight='bold', pad=12)
            plt.xlabel("Number of Top Ensembled Layers (K)", fontsize=11, labelpad=8)
            plt.ylabel("AUROC Score", fontsize=11, labelpad=8)
            plt.xticks(sorted_layers)
            plt.ylim(0.35, 1.02)
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8.5, frameon=True, shadow=True)
            plt.tight_layout()

            chart2_path = os.path.join(fig_dir, f"chart2_layer_depth_sweep_{file_tag}.png")
            plt.savefig(chart2_path, bbox_inches='tight')
            plt.close()
            print(f"  -> Exported Chart 2: {chart2_path}")

        # -------------------------------------------------------------
        # CHART 3: 2D Asymmetric Layer Depth Heatmap (Ka vs Ke)
        # -------------------------------------------------------------
        if em_layers and mi_layers and sorted_layers:
            eval_th = -0.15 if -0.15 in valid_thresholds else valid_thresholds[len(valid_thresholds)//2]
            y_true = (scores_arr < eval_th).astype(int)

            k_vals = sorted_layers
            grid = np.zeros((len(k_vals), len(k_vals)))

            for i, ka in enumerate(k_vals):
                for j, ke in enumerate(k_vals):
                    s_a = (em_layers[ka] - np.mean(em_layers[ka])) / (np.std(em_layers[ka]) + 1e-8)
                    s_e = (mi_layers[ke] - np.mean(mi_layers[ke])) / (np.std(mi_layers[ke]) + 1e-8)
                    joint_score = s_a + s_e
                    grid[i, j] = roc_auc_score(y_true, joint_score)

            best_i, best_j = np.unravel_index(np.argmax(grid), grid.shape)
            best_ka, best_ke = k_vals[best_i], k_vals[best_j]
            best_score = grid[best_i, best_j]

            plt.figure(figsize=(9, 7.5), dpi=150)
            im = plt.imshow(grid, origin='lower', cmap='viridis', aspect='auto')
            plt.colorbar(im, label=f'Joint AU+EU AUROC (Threshold {eval_th})')

            plt.xticks(range(len(k_vals)), k_vals)
            plt.yticks(range(len(k_vals)), k_vals)
            plt.xlabel("Epistemic Layer Depth ($K_e$)", fontsize=11, labelpad=8)
            plt.ylabel("Aleatoric Layer Depth ($K_a$)", fontsize=11, labelpad=8)

            plt.plot(range(len(k_vals)), range(len(k_vals)), color='white', linestyle='--', linewidth=1.5, label='Symmetric ($K_a=K_e$)')
            plt.scatter([best_j], [best_i], color='#e74c3c', s=180, marker='*', zorder=5, label=f'Optimum ($K_a^*={best_ka}, K_e^*={best_ke}$: {best_score:.4f})')

            plt.title(f"Asymmetric $(K_a, K_e)$ Ensembling Heatmap: {run_name}", fontsize=13, fontweight='bold', pad=12)
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9.5, frameon=True)
            plt.tight_layout()

            chart3_path = os.path.join(fig_dir, f"chart3_asymmetric_heatmap_{file_tag}.png")
            plt.savefig(chart3_path, bbox_inches='tight')
            plt.close()
            print(f"  -> Exported Chart 3: {chart3_path}")

        # -------------------------------------------------------------
        # Collect Summary Stats for Tabular Report
        # -------------------------------------------------------------
        std_th = -0.15 if -0.15 in valid_thresholds else valid_thresholds[len(valid_thresholds)//2]
        y_std = (scores_arr < std_th).astype(int)

        summary_rows.append({
            'Model': run['model'],
            'Dataset': run['dataset'],
            'Samples': len(df),
            'Eval_Threshold': std_th,
            'AUROC_entropy_mixture': roc_auc_score(y_std, df['entropy_mixture'].values),
            'AUROC_mutual_info': roc_auc_score(y_std, df['mutual_info'].values),
            'AUROC_entropy_final': roc_auc_score(y_std, df['entropy_final'].values),
            'AUROC_SDLG_Baseline': roc_auc_score(y_std, df['SDLG_Baseline_SE'].values),
            'AUROC_Our_Weighted_SE': roc_auc_score(y_std, df['Our_Weighted_SE'].values),
            'AUROC_Kuhn_SE': roc_auc_score(y_std, df['Kuhn_SE'].values) if df['Kuhn_SE'].notna().any() else np.nan,
            'Optimal_Ka': best_ka if sorted_layers else 'N/A',
            'Optimal_Ke': best_ke if sorted_layers else 'N/A',
            'Optimal_Asym_AUROC': best_score if sorted_layers else np.nan,
        })

    # Save Tabular Summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_out_path = os.path.join(fig_dir, "hyperparameter_evaluation_summary.csv")
        summary_df.to_csv(csv_out_path, index=False)
        print("\n" + "=" * 80)
        print(f"Exported Summary Table to: {csv_out_path}")
        print("=" * 80)
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    export_hyperparameter_charts()
