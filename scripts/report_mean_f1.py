import os, sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from orgpackage.aux import load_experiments

RESULTS_DIR = "./results"
DOMAINS = ["medical", "administrative", "education"]

print("Loading experiments...")
exps = load_experiments()
emb = exps[exps["Technique"] == "embedding"]
sim_exps = emb[emb["Method"] == "similarity"]
clf_exps = emb[emb["Method"] == "classifier"]

# Build lookup: (model, domain, config_key) -> F1
exp_lookup = {}
all_models = set()
all_configs = ["sim_0_shot", "sim_1_shot", "sim_few_shot", "clf_lr", "clf_svm"]
config_labels = {
    "sim_0_shot": "Sim (0-shot)",
    "sim_1_shot": "Sim (1-shot)",
    "sim_few_shot": "Sim (few-shot)",
    "clf_lr": "Clf (LR)",
    "clf_svm": "Clf (SVM)",
}

for _, r in sim_exps.iterrows():
    p = r["Parameters"]
    if isinstance(p, dict):
        model = p.get("model")
        n_shot = p.get("n_shot")
        key = f"sim_{n_shot}"
        exp_lookup[(model, r["Domain"], key)] = float(r["F1"])
        all_models.add(model)

for _, r in clf_exps.iterrows():
    p = r["Parameters"]
    if isinstance(p, dict):
        model = p.get("model")
        head = "lr" if "solver" in p else ("svm" if "kernel" in p else "unknown")
        key = f"clf_{head}"
        exp_lookup[(model, r["Domain"], key)] = float(r["F1"])
        all_models.add(model)

all_models = sorted(list(all_models))

# 1. Model Averages
model_rows = []
for model in all_models:
    f1s = [exp_lookup.get((model, d, c)) for d in DOMAINS for c in all_configs if (model, d, c) in exp_lookup]
    if f1s:
        model_rows.append({"Model": model, "Mean F1": np.mean(f1s)})
model_df = pd.DataFrame(model_rows).sort_values("Mean F1", ascending=False)

# 2. Technique Averages
tech_rows = []
for config in all_configs:
    label = config_labels[config]
    f1s = [exp_lookup.get((m, d, config)) for m in all_models for d in DOMAINS if (m, d, config) in exp_lookup]
    if f1s:
        tech_rows.append({"Technique": label, "Mean F1": np.mean(f1s)})
tech_df = pd.DataFrame(tech_rows).sort_values("Mean F1", ascending=False)

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

latex_models = build_latex_table(model_df, "Model", "Mean F1 per model across all configurations and domains.", "embedding_f1_models")
latex_techs = build_latex_table(tech_df, "Technique", "Mean F1 per technique across all models and domains.", "embedding_f1_techniques")

combined_latex = latex_models + "\n\n" + latex_techs
print(combined_latex)

with open(os.path.join(RESULTS_DIR, "embedding_results_table.tex"), "w") as f:
    f.write(combined_latex)
