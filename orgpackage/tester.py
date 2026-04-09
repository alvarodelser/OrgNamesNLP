from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from orgpackage.aux import load_experiments
from orgpackage.config import DOMAIN_CLASSES_CORR
from orgpackage.evaluator import evaluate_rule_experiment

import re
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


def build_continuous_correctness_table(
    domain: str,
    nli_df: pd.DataFrame,
    experiment_prefix: str,
) -> pd.DataFrame:
    """
    Build a per-entity continuous correctness table for NLI predictions.
    
    Returns a DataFrame: rows = instances (entity URIs),
    columns = experiment IDs, values in [0, 1].
    
    A continuous score for each class is calculated as:
      1 - abs(y_true - conf)
    The instance score is the average of these class scores.
    """
    classes = DOMAIN_CLASSES_CORR[domain]
    
    exp_ids = set()
    for col in nli_df.columns:
        if col.startswith(experiment_prefix) and col.endswith("_conf"):
            for cls in classes:
                suffix = f"_{cls}_conf"
                if col.endswith(suffix):
                    exp_id = col[:-len(suffix)]
                    exp_ids.add(exp_id)
                    break
    
    exp_ids = sorted(list(exp_ids))
    if not exp_ids:
        raise ValueError(f"No experiments found matching prefix {experiment_prefix!r}")
        
    y_true = nli_df.set_index("instance")[classes].values
    
    correctness = {}
    for exp_id in exp_ids:
        conf_cols = [f"{exp_id}_{cls}_conf" for cls in classes]
        missing = [c for c in conf_cols if c not in nli_df.columns]
        if missing:
            continue
            
        y_conf = nli_df.set_index("instance")[conf_cols].values
        
        # Soft accuracy: 1 - absolute error
        scores = 1.0 - np.abs(y_true - y_conf)
        instance_scores = scores.mean(axis=1)
        
        correctness[exp_id] = instance_scores
        
    return pd.DataFrame(correctness, index=nli_df.set_index("instance").index)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Permutation tests
# ══════════════════════════════════════════════════════════════════════════════

def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Simple paired permutation test between two arrays of correctness values.
    Sign-flips the per-entity differences under the null.

    Parameters
    ----------
    a, b : np.ndarray
        Per-entity correctness arrays (values in {0, 1}), same length.

    Returns
    -------
    (observed_difference, two-sided p_value)
    """
    rng = np.random.default_rng(random_state)
    p_min = 1.0 / n_perm

    diff = a - b
    obs = diff.mean()

    signs = rng.choice(np.array([-1, 1]), size=(n_perm, len(diff)))
    perm_stats = (signs * diff).mean(axis=1)

    p_val = max((np.abs(perm_stats) >= abs(obs)).mean(), p_min)
    return obs, p_val


def run_permutation_tests(
    correctness: pd.DataFrame,
    pairs: list[tuple[str, str]],
    n_perm: int = 10_000,
    random_state: int = 42,
    correction: str = "holm",
) -> pd.DataFrame:
    """
    Run permutation tests for a list of experiment pairs.
    Parameters
    ----------
    correctness : pd.DataFrame
        Correctness table with experiment IDs as columns.
    pairs : list of tuple[str, str]
        List of experiment pairs to test.
    n_perm : int
        Number of permutations.
    random_state : int
        Random state.
    correction : str
        Correction method.
    Returns
    -------
    DataFrame with columns: exp_a, exp_b, obs_diff, p_value, p_corrected, significant.
    """
    rows = []
    for exp_a, exp_b in pairs:
        obs, p_val = paired_permutation_test(
            correctness[exp_a].to_numpy(),
            correctness[exp_b].to_numpy(),
            n_perm=n_perm,
            random_state=random_state,
        )
        rows.append({"exp_a": exp_a, "exp_b": exp_b, "obs_diff": obs, "p_value": p_val})

    df = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    reject, p_corrected, _, _ = multipletests(df["p_value"], method=correction)
    df["p_corrected"] = p_corrected
    df["significant"] = reject
    return df



def stratified_permutation_test(
    strata: List[pd.DataFrame],
    exp_a: str,
    exp_b: str,
    n_perm: int = 10_000,
    random_state: int = 42,
    statistic: str = "pooled",
) -> Tuple[float, float]:
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
    strata: List[pd.DataFrame],
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
    correctness: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    title: str = "Pairwise permutation tests",
    alpha: float = 0.05,
    bonferroni: bool = True,
    exp_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
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
        figsize = (7,7)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.RdBu
    cmap.set_bad(color="white")
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    im = ax.imshow(colour_data, cmap=cmap, norm=norm, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Mean correctness diff (A − B)  [coloured only if significant]",
        fontsize=12,
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(exp_order, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(exp_order, fontsize=12)

    p_min = results["p_value"].min()
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=12, color="#aaaaaa")
                continue

            p = p_vals[i, j]
            d = d_vals[i, j]
            sig = bool(sig_mask[i, j])

            p_str = f"p<{p_min:.4f}" if p <= p_min else f"p={p:.3f}"
            label = f"{d:+.2f}\n{p_str}"

            kwargs = dict(
                ha="center", va="center", fontsize=10,
                color="black" if sig else "#999999",
                fontweight="bold" if sig else "normal",
            )
            if sig:
                kwargs["bbox"] = dict(
                    boxstyle="round,pad=0.15", fc="white",
                    ec="black", lw=0.7, alpha=0.6,
                )
            ax.text(j, i, label, **kwargs)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig




# ----------------------------------------------------------------------
# Word-level coverage utilities
# ----------------------------------------------------------------------
def compute_word_coverage_for_experiment(exp_row, test_df):
    """
    Compute per-word TP / FP counts per country for a single experiment row.

    Returns a DataFrame with columns:
        ['exp_id','domain','cls','country','word','tp','fp','total']
    """
    domain = exp_row["Domain"]
    exp_id = exp_row["ID"]
    classes = DOMAIN_CLASSES_CORR[domain]

    # keywords dict: {'whitelist_hospital': {country: [w1, w2, ...]}, ...}
    kw_dict = exp_row["Parameters"]["keywords"]

    rows = []

    for cls in classes:
        whitelist_key = f"whitelist_{cls}"
        if whitelist_key not in kw_dict:
            continue

        country_to_words = kw_dict[whitelist_key]  # {country: [words]}

        for country, words in country_to_words.items():
            if not words:
                continue

            df_country = test_df[test_df["country"] == country]
            if df_country.empty:
                continue

            true_pos_mask = df_country[cls] == 1

            # For each word, compute TP / FP in this country for this class
            for word in words:
                pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
                name_matches = df_country["names"].astype(str).apply(
                    lambda x: bool(pattern.search(x))
                )

                tp = (name_matches & true_pos_mask).sum()
                fp = (name_matches & ~true_pos_mask).sum()
                total = tp + fp

                if total == 0:
                    continue

                rows.append(
                    {
                        "exp_id": exp_id,
                        "domain": domain,
                        "cls": cls,
                        "country": country,
                        "word": word,
                        "tp": int(tp),
                        "fp": int(fp),
                        "total": int(total),
                    }
                )

    return pd.DataFrame(rows)


def build_word_coverage_table(experiments_df, tests, exp_ids_of_interest):
    """
    Build a coverage_df table for the selected experiments, matching the shape
    previously created in the notebook.
    """
    all_coverage = []

    rule_idf_exps = experiments_df[
        (experiments_df["Technique"] == "rules")
        & (experiments_df["Method"] == "idf_best")
        & (experiments_df["ID"].isin(exp_ids_of_interest))
    ].reset_index(drop=True)

    for _, exp_row in rule_idf_exps.iterrows():
        domain = exp_row["Domain"]
        if domain not in tests:
            raise ValueError(
                f"`tests` dict with key '{domain}' not found. "
                f"Make sure you have something like tests['{domain}'] defined."
            )
        test_df = tests[domain]
        cov_df = compute_word_coverage_for_experiment(exp_row, test_df)
        all_coverage.append(cov_df)

    if not all_coverage:
        return pd.DataFrame()

    return pd.concat(all_coverage, ignore_index=True)
