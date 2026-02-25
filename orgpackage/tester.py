from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from orgpackage.aux import load_experiments
from orgpackage.config import DOMAIN_CLASSES_CORR
from orgpackage.evaluator import evaluate_rule_experiment


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Per-entity correctness tables
# ══════════════════════════════════════════════════════════════════════════════

def build_correctness_table(
    domain: str,
    test_df: pd.DataFrame,
    technique: str = "rules",
    experiments_path: str = "./results/experiments.csv",
) -> pd.DataFrame:
    """
    Build a per-entity exact-match correctness table for one domain.

    Returns a DataFrame: rows = instances (entity URIs),
    columns = experiment IDs, values in {0, 1}.
    1 means the experiment's multi-label prediction vector matches the
    ground-truth label vector for that entity.

    Uses evaluate_rule_experiment from evaluator.py to generate predictions
    (reuses existing code, no duplication).
    """
    exps = load_experiments(experiments_path)
    domain_exps = exps[
        (exps["Domain"] == domain) & (exps["Technique"] == technique)
    ].reset_index(drop=True)

    if domain_exps.empty:
        raise ValueError(
            f"No experiments found for domain={domain!r}, technique={technique!r}"
        )

    df = test_df.copy()
    classes = DOMAIN_CLASSES_CORR[domain]

    # Run classification for each experiment (adds <exp_id>_<cls> columns)
    if technique == "rules":
        for _, exp_row in domain_exps.iterrows():
            evaluate_rule_experiment(exp_row, df, test_countries=None)

    pred_cols_per_exp = {}
    for exp_id in domain_exps["ID"]:
        cols = [f"{exp_id}_{cls}" for cls in classes if f"{exp_id}_{cls}" in df.columns]
        if cols:
            pred_cols_per_exp[exp_id] = cols

    if not pred_cols_per_exp:
        raise ValueError("No prediction columns were generated.")

    y_true = df.set_index("instance")[classes].values
    correctness = {}
    for exp_id, pred_cols in pred_cols_per_exp.items():
        y_pred = df.set_index("instance")[pred_cols].values
        correctness[exp_id] = (y_pred == y_true).all(axis=1).astype(int)

    return pd.DataFrame(correctness, index=df.set_index("instance").index)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Permutation tests
# ══════════════════════════════════════════════════════════════════════════════

def stratified_permutation_test(
    strata: list[pd.DataFrame],
    exp_a: str,
    exp_b: str,
    n_perm: int = 10_000,
    random_state: int = 42,
    statistic: str = "pooled",
) -> tuple[float, float]:
    """
    Stratified (blocked) paired permutation test between two experiments.

    Parameters
    ----------
    strata : list of DataFrames
        Each element is a per-entity correctness table (rows=entities,
        columns=experiment IDs, values 0/1).  Each table is one stratum
        (e.g. one domain).  All tables must contain both ``exp_a`` and
        ``exp_b`` as columns.  The strata can have different numbers of
        rows (entities), but there must be at least two strata-level
        lists of the same length — this is automatically satisfied
        because each stratum is a self-contained DataFrame.
    exp_a, exp_b : str
        Column names identifying the two experiments to compare.
    n_perm : int
        Number of permutations.
    statistic : str
        ``"pooled"`` — pool all entities across strata (weighted by
        stratum size).
        ``"mean_of_strata"`` — average the per-stratum means (equal
        weight per stratum).

    Returns
    -------
    (observed_difference, p_value)
        Two-sided p-value, clamped to a minimum of 1/n_perm.
    """
    rng = np.random.default_rng(random_state)
    p_min = 1.0 / n_perm

    for i, s in enumerate(strata):
        if exp_a not in s.columns or exp_b not in s.columns:
            raise ValueError(
                f"Stratum {i} is missing column {exp_a!r} or {exp_b!r}. "
                f"Available: {list(s.columns)}"
            )

    diffs = [(s[exp_a] - s[exp_b]).to_numpy() for s in strata]

    # ── observed statistic ────────────────────────────────────────────────
    if statistic == "pooled":
        obs = np.concatenate(diffs).mean()
    else:
        obs = np.mean([d.mean() for d in diffs])

    # ── null distribution (sign-flip within each stratum) ─────────────────
    perm_stats = np.zeros(n_perm)
    for diff_arr in diffs:
        signs = rng.choice(np.array([-1, 1]), size=(n_perm, len(diff_arr)))
        contrib = signs * diff_arr
        if statistic == "pooled":
            perm_stats += contrib.sum(axis=1)
        else:
            perm_stats += contrib.mean(axis=1)

    if statistic == "pooled":
        perm_stats /= sum(len(d) for d in diffs)
    else:
        perm_stats /= len(diffs)

    p_val = max((np.abs(perm_stats) >= abs(obs)).mean(), p_min)
    return obs, p_val


def run_all_pairwise_tests(
    strata: list[pd.DataFrame],
    n_perm: int = 10_000,
    random_state: int = 42,
    statistic: str = "pooled",
) -> pd.DataFrame:
    """
    Run stratified_permutation_test for every experiment pair.

    When ``strata`` is a single-element list this is equivalent to a
    standard (non-stratified) paired permutation test.

    Parameters
    ----------
    strata : list of DataFrames
        Same format as ``stratified_permutation_test``.
        Only experiments (columns) present in **all** strata are tested.

    Returns
    -------
    DataFrame with columns: exp_a, exp_b, obs_diff, p_value.
    """
    common_exps = sorted(
        set.intersection(*(set(s.columns) for s in strata))
    )
    if len(common_exps) < 2:
        raise ValueError(
            f"Need ≥2 common experiments across strata, found {len(common_exps)}"
        )

    rows = []
    for exp_a, exp_b in combinations(common_exps, 2):
        obs, p_val = stratified_permutation_test(
            strata, exp_a, exp_b,
            n_perm=n_perm,
            random_state=random_state,
            statistic=statistic,
        )
        rows.append({
            "exp_a": exp_a, "exp_b": exp_b,
            "obs_diff": obs, "p_value": p_val,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_permutation_heatmap(
    results: pd.DataFrame,
    correctness: pd.DataFrame | list[pd.DataFrame] | None = None,
    title: str = "Pairwise permutation tests",
    alpha: float = 0.05,
    bonferroni: bool = True,
    exp_order: list[str] | None = None,
    save_path: str | None = None,
    figsize: tuple | None = None,
):
    """
    Plot a symmetric heatmap of pairwise permutation test results.

    Parameters
    ----------
    results : DataFrame
        Output of ``run_all_pairwise_tests``.
    correctness : DataFrame or list of DataFrames, optional
        Used only to sort experiments by descending mean correctness when
        ``exp_order`` is not provided.  If a list, rows are concatenated.
    title : str
        Plot title.
    alpha : float
        Significance level before correction.
    bonferroni : bool
        Apply Bonferroni correction.
    exp_order : list of str, optional
        Custom experiment ordering for axes.  If None, experiments are
        ordered by descending mean correctness (requires ``correctness``).
    save_path : str, optional
        If provided the figure is saved to this path.
    figsize : tuple, optional
        Matplotlib figure size.  Auto-sized if None.
    """
    # ── experiment ordering ───────────────────────────────────────────────
    all_exps = sorted(
        set(results["exp_a"]).union(results["exp_b"])
    )
    if exp_order is None:
        if correctness is not None:
            if isinstance(correctness, list):
                pooled = pd.concat(correctness)
            else:
                pooled = correctness
            mean_c = pooled[all_exps].mean().sort_values(ascending=False)
            exp_order = mean_c.index.tolist()
        else:
            exp_order = all_exps

    n = len(exp_order)
    n_tests = len(results)
    alpha_adj = alpha / n_tests if bonferroni else alpha

    # ── build symmetric matrices ──────────────────────────────────────────
    idx_map = {e: i for i, e in enumerate(exp_order)}
    p_vals = np.ones((n, n))
    d_vals = np.zeros((n, n))

    for _, row in results.iterrows():
        a, b = row["exp_a"], row["exp_b"]
        if a not in idx_map or b not in idx_map:
            continue
        i, j = idx_map[a], idx_map[b]
        p_vals[i, j] = p_vals[j, i] = row["p_value"]
        d_vals[i, j] = row["obs_diff"]
        d_vals[j, i] = -row["obs_diff"]

    np.fill_diagonal(p_vals, np.nan)
    np.fill_diagonal(d_vals, np.nan)

    sig_mask = p_vals <= alpha_adj
    colour_data = np.where(sig_mask, d_vals, np.nan)

    vlim = max(np.nanmax(np.abs(d_vals)), 1e-6)

    # ── plot ──────────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (max(8, n * 0.75), max(7, n * 0.75))

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.RdBu
    cmap.set_bad(color="#e0e0e0")
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    im = ax.imshow(colour_data, cmap=cmap, norm=norm, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Mean correctness diff (A − B)  [coloured only if significant]",
        fontsize=9,
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(exp_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(exp_order, fontsize=8)

    p_min = results["p_value"].min()
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#aaaaaa")
                continue

            p = p_vals[i, j]
            d = d_vals[i, j]
            sig = bool(sig_mask[i, j])

            p_str = f"p<{p_min:.4f}" if p <= p_min else f"p={p:.3f}"
            label = f"{d:+.2f}\n{p_str}"

            kwargs = dict(
                ha="center", va="center", fontsize=6,
                color="black" if sig else "#999999",
                fontweight="bold" if sig else "normal",
            )
            if sig:
                kwargs["bbox"] = dict(
                    boxstyle="round,pad=0.15", fc="white",
                    ec="black", lw=0.7, alpha=0.6,
                )
            ax.text(j, i, label, **kwargs)

    corr_label = f"(Bonf.) " if bonferroni else ""
    ax.set_title(
        f"{title}\n"
        f"Colour = mean correctness diff (row − col), "
        f"blue = row better, red = col better\n"
        f"Bold + box = significant  (α{corr_label}= {alpha_adj:.4f})",
        fontsize=10, pad=12,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig
