"""
run_embedding_permutation_analysis.py
======================================
Two stratified permutation tests on embedding experiments:

TEST 1 — MODEL COMPARISON
    Compares models pairwise. For each model pair, strata are all matched
    experiments where domain + method + config are identical, only the
    embedding model differs. Heatmap: model × model.

TEST 2 — TECHNIQUE COMPARISON
    5 treatments: Similarity(0-shot), Similarity(1-shot), Similarity(few-shot),
    Classifier(LR), Classifier(SVM).
    For each pair, strata are all (model, domain) combos where both treatments
    exist. Heatmap: technique × technique.

Global Holm-Bonferroni correction applied across all tests.

Run:  python run_embedding_permutation_analysis.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import combinations
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from orgpackage.aux import load_experiments
from orgpackage.config import DOMAIN_CLASSES_CORR
from orgpackage.tester import stratified_permutation_test

warnings.filterwarnings("ignore")

CORRECTNESS_DIR = "./results/correctness_tables"
FIGURES_DIR     = "./figures"
RESULTS_DIR     = "./results"
DOMAINS         = ["medical", "administrative", "education"]
N_PERM          = 2_000

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("Loading experiments…")
exps = load_experiments()
emb  = exps[exps["Technique"] == "embedding"]
sim_exps = emb[emb["Method"] == "similarity"]
clf_exps = emb[emb["Method"] == "classifier"]

# Build lookup: (model, domain, config_key) → experiment_id
#   config_key = "sim_0_shot" / "sim_1_shot" / "sim_few_shot" / "clf_lr" / "clf_svm"
exp_lookup = {}  # (model, domain, config_key) → experiment_id

for _, r in sim_exps.iterrows():
    p = r["Parameters"]
    if not isinstance(p, dict):
        continue
    model   = p.get("model")
    n_shot  = p.get("n_shot")
    domain  = r["Domain"]
    key     = f"sim_{n_shot}"
    exp_lookup[(model, domain, key)] = r["ID"]

for _, r in clf_exps.iterrows():
    p = r["Parameters"]
    if not isinstance(p, dict):
        continue
    model  = p.get("model")
    head   = "lr" if "solver" in p else ("svm" if "kernel" in p else "unknown")
    domain = r["Domain"]
    key    = f"clf_{head}"
    exp_lookup[(model, domain, key)] = r["ID"]

# Load correctness tables
print("Loading correctness tables…")
sim_ct = {}; clf_ct = {}
for f in os.listdir(CORRECTNESS_DIR):
    if not f.endswith(".csv"):
        continue
    name = f.replace(".csv", "")
    path = os.path.join(CORRECTNESS_DIR, f)
    if name.endswith("_sim"):
        m = name[:-4]; sim_ct[m] = pd.read_csv(path, index_col=0)
        sim_ct[m] = sim_ct[m][~sim_ct[m].index.duplicated(keep='first')]
    elif name.endswith("_clf"):
        m = name[:-4]; clf_ct[m] = pd.read_csv(path, index_col=0)
        clf_ct[m] = clf_ct[m][~clf_ct[m].index.duplicated(keep='first')]

print(f"  Similarity tables: {sorted(sim_ct.keys())}")
print(f"  Classifier tables: {sorted(clf_ct.keys())}")

all_models = sorted(set(list(sim_ct.keys()) + list(clf_ct.keys())))
all_configs = ["sim_0_shot", "sim_1_shot", "sim_few_shot", "clf_lr", "clf_svm"]
config_labels = {
    "sim_0_shot":  "Sim (0-shot)",
    "sim_1_shot":  "Sim (1-shot)",
    "sim_few_shot": "Sim (few-shot)",
    "clf_lr":      "Clf (LR)",
    "clf_svm":     "Clf (SVM)",
}


def get_correctness(model, domain, config_key):
    """Retrieve the correctness Series for a (model, domain, config) triple."""
    eid = exp_lookup.get((model, domain, config_key))
    if eid is None:
        return None
    if config_key.startswith("sim_"):
        ct = sim_ct.get(model)
    else:
        ct = clf_ct.get(model)
    if ct is None or eid not in ct.columns:
        return None
    return ct[eid]


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def make_heatmap(pw_df, title, save_name, exp_order, mean_acc=None, figsize=(9, 8)):
    """
    Full symmetric heatmap. Colored cells = significant after Bonferroni.
    mean_acc: optional dict {label: mean_accuracy} for ordering.
    """
    if pw_df.empty:
        print(f"  [SKIP] No data for {title}")
        return

    n = len(exp_order)
    n_tests = len(pw_df)
    alpha_adj = 0.05 / n_tests if n_tests > 0 else 0.05

    idx_map = {e: i for i, e in enumerate(exp_order)}
    p_vals = np.ones((n, n))
    d_vals = np.zeros((n, n))

    for _, row in pw_df.iterrows():
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

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="white")
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    im = ax.imshow(colour_data, cmap=cmap, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean correctness diff (row − col)  [coloured = significant]", fontsize=11)

    # Labels: optionally show mean accuracy
    if mean_acc:
        labels = [f"{e}\n(acc={mean_acc.get(e, 0):.3f})" for e in exp_order]
    else:
        labels = exp_order

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    p_min = pw_df["p_value"].min() if not pw_df.empty else 1e-4
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="#aaaaaa")
                continue
            p = p_vals[i, j]
            d = d_vals[i, j]
            sig = bool(sig_mask[i, j])
            p_str = f"p<{p_min:.4f}" if p <= p_min else f"p={p:.3f}"
            label_txt = f"{d:+.3f}\n{p_str}"
            kwargs = dict(ha="center", va="center", fontsize=9,
                          color="black" if sig else "#999999",
                          fontweight="bold" if sig else "normal")
            if sig:
                kwargs["bbox"] = dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="black", lw=0.7, alpha=0.6)
            ax.text(j, i, label_txt, **kwargs)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: MODEL COMPARISON
#
# For each pair (model_a, model_b), build strata from every matched
# (domain, config) where both models have data. Each match = one stratum.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: MODEL COMPARISON")
print("  Strata = matched (domain, config) pairs per model pair")
print("=" * 70)

pw_model_rows = []
for ma, mb in combinations(all_models, 2):
    strata = []
    for config in all_configs:
        for domain in DOMAINS:
            sa = get_correctness(ma, domain, config)
            sb = get_correctness(mb, domain, config)
            if sa is not None and sb is not None:
                # Join on instance index to get matched pairs
                joined = pd.DataFrame({"A": sa, "B": sb}).dropna()
                if len(joined) > 0:
                    strata.append(joined)

    if not strata:
        print(f"  {ma} vs {mb}: no matched strata")
        continue

    obs, pval = stratified_permutation_test(
        strata, "A", "B", n_perm=N_PERM, random_state=42, statistic="pooled"
    )
    n_strata = len(strata)
    n_instances = sum(len(s) for s in strata)
    pw_model_rows.append({
        "exp_a": ma, "exp_b": mb,
        "obs_diff": obs, "p_value": pval,
    })
    print(f"  {ma} vs {mb}: diff={obs:+.4f}, p={pval:.4f}  ({n_strata} strata, {n_instances:,} instances)")

pw_models = pd.DataFrame(pw_model_rows)

# Compute mean accuracy per model (for ordering)
model_acc = {}
for m in all_models:
    vals = []
    for config in all_configs:
        for domain in DOMAINS:
            s = get_correctness(m, domain, config)
            if s is not None:
                vals.append(s.mean())
    if vals:
        model_acc[m] = np.mean(vals)

# Order by descending accuracy
model_order = sorted(all_models, key=lambda m: -model_acc.get(m, 0))

make_heatmap(pw_models,
             "Embedding Models: Pairwise Permutation Tests",
             "heatmap_models.png",
             exp_order=model_order,
             mean_acc=model_acc,
             figsize=(10, 9))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: TECHNIQUE COMPARISON
#
# 5 treatments: Sim(0), Sim(1), Sim(few), Clf(LR), Clf(SVM)
# For each pair, strata = every (model, domain) where both treatments exist.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: TECHNIQUE COMPARISON")
print("  Strata = matched (model, domain) pairs per technique pair")
print("=" * 70)

pw_tech_rows = []
for ca, cb in combinations(all_configs, 2):
    la, lb = config_labels[ca], config_labels[cb]
    strata = []
    for model in all_models:
        for domain in DOMAINS:
            sa = get_correctness(model, domain, ca)
            sb = get_correctness(model, domain, cb)
            if sa is not None and sb is not None:
                joined = pd.DataFrame({"A": sa, "B": sb}).dropna()
                if len(joined) > 0:
                    strata.append(joined)

    if not strata:
        print(f"  {la} vs {lb}: no matched strata")
        continue

    obs, pval = stratified_permutation_test(
        strata, "A", "B", n_perm=N_PERM, random_state=42, statistic="pooled"
    )
    n_strata = len(strata)
    n_instances = sum(len(s) for s in strata)
    pw_tech_rows.append({
        "exp_a": la, "exp_b": lb,
        "obs_diff": obs, "p_value": pval,
    })
    print(f"  {la} vs {lb}: diff={obs:+.4f}, p={pval:.4f}  ({n_strata} strata, {n_instances:,} instances)")

pw_techs = pd.DataFrame(pw_tech_rows)

# Compute mean accuracy per technique
tech_acc = {}
for config in all_configs:
    label = config_labels[config]
    vals = []
    for model in all_models:
        for domain in DOMAINS:
            s = get_correctness(model, domain, config)
            if s is not None:
                vals.append(s.mean())
    if vals:
        tech_acc[label] = np.mean(vals)

tech_order = sorted([config_labels[c] for c in all_configs],
                    key=lambda t: -tech_acc.get(t, 0))

make_heatmap(pw_techs,
             "Technique Comparison: Pairwise Permutation Tests",
             "heatmap_techniques.png",
             exp_order=tech_order,
             mean_acc=tech_acc,
             figsize=(10, 9))


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL HOLM CORRECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GLOBAL HOLM-BONFERRONI CORRECTION")
print("=" * 70)

all_rows = []
for _, r in pw_models.iterrows():
    all_rows.append({"analysis": "T1_model", "comparison": f"{r['exp_a']} vs {r['exp_b']}",
                     "obs_diff": r["obs_diff"], "p_value": r["p_value"]})
for _, r in pw_techs.iterrows():
    all_rows.append({"analysis": "T2_technique", "comparison": f"{r['exp_a']} vs {r['exp_b']}",
                     "obs_diff": r["obs_diff"], "p_value": r["p_value"]})

results_df = pd.DataFrame(all_rows)
if not results_df.empty:
    rej, pc, _, _ = multipletests(results_df["p_value"], method="holm")
    results_df["p_corrected"] = pc
    results_df["significant"] = rej
    results_df.to_csv(os.path.join(RESULTS_DIR, "embedding_permutation_tests.csv"), index=False)
    print(f"  {rej.sum()} / {len(rej)} significant after Holm correction")
    print()
    print(results_df[["analysis", "comparison", "obs_diff", "p_value", "p_corrected", "significant"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES — Mean F1 per Model and per Technique
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING LATEX TABLES")
print("=" * 70)

all_emb = pd.concat([sim_exps, clf_exps])

# 1. Table Data: Model Averages
model_rows = []
for model in sorted(all_models):
    f1s = []
    for config in all_configs:
        for domain in DOMAINS:
            eid = exp_lookup.get((model, domain, config))
            if eid:
                match = all_emb[all_emb["ID"] == eid]
                if not match.empty:
                    f1s.append(float(match["F1"].iloc[0]))
    if f1s:
        model_rows.append({"Model": model, "Mean F1": np.mean(f1s)})

model_df = pd.DataFrame(model_rows).sort_values("Mean F1", ascending=False)

# 2. Table Data: Technique Averages
tech_rows = []
for config in all_configs:
    f1s = []
    label = config_labels[config]
    for model in all_models:
        for domain in DOMAINS:
            eid = exp_lookup.get((model, domain, config))
            if eid:
                match = all_emb[all_emb["ID"] == eid]
                if not match.empty:
                    f1s.append(float(match["F1"].iloc[0]))
    if f1s:
        tech_rows.append({"Technique": label, "Mean F1": np.mean(f1s)})

tech_df = pd.DataFrame(tech_rows).sort_values("Mean F1", ascending=False)

# Store summarized CSVs
model_df.to_csv(os.path.join(RESULTS_DIR, "embedding_summary_models.csv"), index=False)
tech_df.to_csv(os.path.join(RESULTS_DIR, "embedding_summary_techniques.csv"), index=False)

def build_latex_table(df, label_col, caption, label_ref):
    best_f1 = df["Mean F1"].max()
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{tab:{label_ref}}}")
    latex.append(r"\begin{tabular}{lr}")
    latex.append(r"\toprule")
    latex.append(f"\\textbf{{{label_col}}} & \\textbf{{Mean F1}} \\\\")
    latex.append(r"\midrule")
    for _, r in df.iterrows():
        f1_str = f"\\textbf{{{r['Mean F1']:.4f}}}" if abs(r["Mean F1"] - best_f1) < 1e-6 else f"{r['Mean F1']:.4f}"
        latex.append(f"{r[label_col]} & {f1_str} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

# Generate both tables
latex_models = build_latex_table(model_df, "Model", "Mean F1 per model across all configurations and domains.", "embedding_f1_models")
latex_techs = build_latex_table(tech_df, "Technique", "Mean F1 per technique across all models and domains.", "embedding_f1_techniques")

combined_latex = latex_models + "\n\n" + latex_techs
print(combined_latex)

latex_path = os.path.join(RESULTS_DIR, "embedding_results_table.tex")
with open(latex_path, "w") as f:
    f.write(combined_latex)
print(f"\nLaTeX tables saved to: {latex_path}")

print("\n" + "=" * 70)
print("DONE — heatmaps in figures/, results in results/")
print("=" * 70)
