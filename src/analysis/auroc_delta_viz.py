import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

DELTA_CMAP = LinearSegmentedColormap.from_list(
    'delta', ['#A32D2D', '#F09595', '#f5f5f0', '#97C459', '#3B6D11'], N=256
)

def _delta_norm(vmax=0.06):
    return Normalize(vmin=-vmax, vmax=vmax)


def plot_auroc_dashboard(
    all_results,
    dataset,
    metric,
    mode='normal',
    models=None,
    correctness_thresholds=None,   # list[float] for this dataset
    density_thresholds=None,
    figsize=None,
    delta_vmax=0.06,
    save_path=None,
):
    """
    One figure for a single dataset, one column per model.

    Parameters
    ----------
    all_results : dict
        all_results[dataset][model][metric][d_thresh][c_thresh]
        Leaf keys: 'normal', 'reversed', 'base_sem', 'base_epist', 'base_new', 'base_avg'
    dataset : str
    metric : str
    mode : str          'normal' or 'reversed'
    models : list[str]  defaults to all keys in all_results[dataset]
    correctness_thresholds : list[float]   thresholds for this dataset
    density_thresholds : list[float]       defaults to keys in all_results
    figsize : tuple
    delta_vmax : float  colour scale clamps at ±delta_vmax
    save_path : str     optional
    """
    if models is None:
        models = list(all_results[dataset].keys())
    n_models = len(models)

    first = all_results[dataset][models[0]][metric]
    if density_thresholds is None:
        density_thresholds = sorted(first.keys())
    if correctness_thresholds is None:
        correctness_thresholds = sorted(first[density_thresholds[0]].keys())

    d_arr = np.array(density_thresholds)
    c_arr = np.array(correctness_thresholds)

    # ── build delta matrix per model ─────────────────────────────────────────
    def _build_delta_matrix(model):
        mat = np.full((len(c_arr), len(d_arr)), np.nan)
        for ci, c in enumerate(c_arr):
            for di, d in enumerate(d_arr):
                try:
                    cell = all_results[dataset][model][metric][d][c]
                    best_base = max(cell['base_sem'], cell['base_epist'],
                                    cell['base_new'], cell['base_avg'])
                    mat[ci, di] = cell[mode] - best_base
                except KeyError:
                    pass
        return mat

    matrices = {m: _build_delta_matrix(m) for m in models}

    # ── best density threshold: per-model and consistent across models ────────
    def _best_d(mat):
        col_means = np.nanmean(mat, axis=0)
        return d_arr[np.argmax(col_means)]

    per_model_best = {m: _best_d(matrices[m]) for m in models}
    combined = np.stack([matrices[m] for m in models], axis=0)   # (n_models, n_c, n_d)
    consistent_best = d_arr[np.argmax(np.nanmean(combined, axis=(0, 1)))]

    # ── figure layout ─────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (5.2 * n_models + 1.5, max(5, 0.55 * len(c_arr) + 3.0))

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#fafaf8')

    fig.suptitle(
        f"Delta vs best baseline   |   dataset: {dataset}   metric: {metric}   mode: {mode}",
        fontsize=12, fontweight='500', y=0.99, color='#2C2C2A'
    )

    norm = _delta_norm(delta_vmax)

    # consistent-best banner
    ax_banner = fig.add_axes([0.02, 0.91, 0.96, 0.045])
    ax_banner.set_axis_off()
    ax_banner.text(
        0.5, 0.5,
        f"Consistent best density threshold across all models:  d = {consistent_best:.2f}"
        f"   (maximises avg Δ vs best baseline)",
        ha='center', va='center', fontsize=10, color='#042C53',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#B5D4F4',
                  edgecolor='#185FA5', linewidth=0.8)
    )

    # ── per-model heatmap panels ───────────────────────────────────────────────
    left_margin  = 0.07
    right_margin = 0.10
    top          = 0.88
    bottom       = 0.12
    panel_w = (1 - left_margin - right_margin) / n_models
    gap     = 0.015

    for i, model in enumerate(models):
        mat    = matrices[model]
        best_d = per_model_best[model]

        ax = fig.add_axes([
            left_margin + i * panel_w + gap / 2,
            bottom, panel_w - gap, top - bottom
        ])

        ax.imshow(mat, cmap=DELTA_CMAP, norm=norm, aspect='auto',
                  interpolation='nearest')

        # cell text
        for ci in range(len(c_arr)):
            for di in range(len(d_arr)):
                val = mat[ci, di]
                if np.isnan(val):
                    continue
                t = norm(val)
                text_col = 'white' if (t < 0.25 or t > 0.75) else '#2C2C2A'
                ax.text(di, ci, f"{val:+.3f}", ha='center', va='center',
                        fontsize=8, color=text_col, fontweight='500')

        # column highlight overlays
        for di, d in enumerate(d_arr):
            is_best = np.isclose(d, best_d)
            is_cons = np.isclose(d, consistent_best)
            if is_best:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (di - 0.48, -0.48), 0.96, len(c_arr) - 0.04,
                    boxstyle="square,pad=0", linewidth=2,
                    edgecolor='#185FA5', facecolor='none', zorder=3
                ))
            elif is_cons:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (di - 0.48, -0.48), 0.96, len(c_arr) - 0.04,
                    boxstyle="square,pad=0", linewidth=1.5,
                    edgecolor='#185FA5', facecolor='none', linestyle='--', zorder=3
                ))

        # x-axis labels (density thresholds)
        x_labels = []
        for d in d_arr:
            label = f"d={d:.2f}"
            if   np.isclose(d, best_d) and np.isclose(d, consistent_best): label += "\n★"
            elif np.isclose(d, best_d):   label += "\n▲"
            elif np.isclose(d, consistent_best): label += "\n◆"
            x_labels.append(label)
        ax.set_xticks(range(len(d_arr)))
        ax.set_xticklabels(x_labels, fontsize=8, color='#5F5E5A')

        # y-axis labels only on leftmost panel
        if i == 0:
            ax.set_yticks(range(len(c_arr)))
            ax.set_yticklabels([f"c={c:.2f}" for c in c_arr], fontsize=8, color='#5F5E5A')
        else:
            ax.set_yticks([])

        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # panel title + per-model summary stats
        avg_best = float(np.nanmean(mat[:, np.where(np.isclose(d_arr, best_d))[0]]))
        avg_cons = float(np.nanmean(mat[:, np.where(np.isclose(d_arr, consistent_best))[0]]))
        ax.set_title(
            f"{model}\nbest d={best_d:.2f}  (avg Δ={avg_best:+.4f})"
            f"\nat d={consistent_best:.2f}  (avg Δ={avg_cons:+.4f})",
            fontsize=9, color='#2C2C2A', pad=6, loc='left', linespacing=1.6
        )

    # ── shared colorbar ───────────────────────────────────────────────────────
    cb_ax = fig.add_axes([1 - right_margin + 0.02, bottom, 0.022, top - bottom])
    sm = ScalarMappable(cmap=DELTA_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("Δ AUROC vs best baseline", fontsize=8, color='#5F5E5A')
    cb.ax.tick_params(labelsize=7, colors='#5F5E5A')
    cb.outline.set_visible(False)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color='#3B6D11', label='strong gain (>+0.03)'),
        mpatches.Patch(color='#97C459', label='gain'),
        mpatches.Patch(color='#f5f5f0', label='neutral', ec='#B4B2A9', linewidth=0.5),
        mpatches.Patch(color='#F09595', label='loss'),
        mpatches.Patch(color='#A32D2D', label='strong loss (<-0.03)'),
        mpatches.Patch(facecolor='none', edgecolor='#185FA5',
                       linewidth=2, label='▲ best d (this model)'),
        mpatches.Patch(facecolor='none', edgecolor='#185FA5',
                       linewidth=1.5, linestyle='--', label='◆ consistent best d'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.45, 0.0),
               labelcolor='#444441')

    plt.subplots_adjust(bottom=0.12)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")

    plt.show()
    return fig


def find_consistent_density_threshold(all_results, dataset, metric,
                                       mode='normal', models=None,
                                       correctness_thresholds=None,
                                       density_thresholds=None):
    """Prints and returns a DataFrame of avg delta per d_thresh, per model + overall."""
    import pandas as pd

    if models is None:
        models = list(all_results[dataset].keys())
    first = all_results[dataset][models[0]][metric]
    if density_thresholds is None:
        density_thresholds = sorted(first.keys())
    if correctness_thresholds is None:
        correctness_thresholds = sorted(first[density_thresholds[0]].keys())

    rows = []
    for d in density_thresholds:
        row = {'d_thresh': d}
        total = 0
        for m in models:
            deltas = []
            for c in correctness_thresholds:
                try:
                    cell = all_results[dataset][m][metric][d][c]
                    best_base = max(cell['base_sem'], cell['base_epist'],
                                    cell['base_new'], cell['base_avg'])
                    deltas.append(cell[mode] - best_base)
                except KeyError:
                    pass
            avg = np.mean(deltas) if deltas else np.nan
            row[f'avg_delta_{m}'] = round(avg, 5)
            total += avg if not np.isnan(avg) else 0
        row['avg_delta_ALL'] = round(total / len(models), 5)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('d_thresh')
    best_d = df['avg_delta_ALL'].idxmax()
    print(f"\nDataset: {dataset}  |  Consistent best density threshold: d = {best_d:.2f}")
    display(df)
    return df, best_d