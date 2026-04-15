"""
build_cd_diagram_data.py
========================
Builds the per-country F1 table needed for Critical Difference (CD) Diagrams.

The output is a CSV with shape:
    rows    = (domain, country) pairs
    columns = technique labels (one comparison system per column)
    values  = macro-F1 score for that (domain, country) pair

Strategy per technique type
---------------------------
RULES  (med-r-idf-4 / adm-r-idf-1 / edu-r-idf-1   and   xxx-r-llm-0)
    Country-level F1 is already stored inside the Parameters["country_f1"]
    dict of experiments_final.csv.  We just extract it.

NLI   (xxx-n-0-2   = mDeBERTa)
    Per-instance predictions are stored in nli_confidences.csv.
    We reconstruct the test set (union of instances the evaluator actually
    saw for each domain) and compute F1 per country from scratch.

EMBEDDINGS  (xxx-e-similarity-N / xxx-e-classifier-N)
    Per-instance exact-match correctness tables are pre-built in
    results/correctness_tables/.  Since only binary correctness (not
    per-class predictions) is stored, we compute per-country mean
    exact-match correctness as the metric.

Run this script incrementally: after adding the next technique group,
re-run – existing columns are preserved and new ones are appended.

Usage
-----
    python build_cd_diagram_data.py
"""

import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ── project root -----------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from orgpackage.aux import load_experiments
from orgpackage.config import DOMAIN_CLASSES_CORR, COUNTRY_DICT

# ── paths ------------------------------------------------------------------------
EXPERIMENTS_FINAL_PATH = "results/experiments_final.csv"
NLI_CONFIDENCES_PATH   = "results/nli_confidences.csv"
DATASET_PATH           = "data/wikidata_enriched_dataset.csv"
TOKENIZED_PATH         = "results/tokenized_names.csv"
DECOMPOSED_PATH        = "results/decomposed_names.csv"
OUTPUT_PATH            = "results/cd_diagram_data.csv"
CORRECTNESS_DIR        = "results/correctness_tables"

# ── domain prefix helpers --------------------------------------------------------
DOMAIN_PREFIX = {
    "medical":        "med",
    "administrative": "adm",
    "education":      "edu",
}
PREFIX_DOMAIN = {v: k for k, v in DOMAIN_PREFIX.items()}

# ── minimum country volume to include (same threshold used in evaluator) ---------
MIN_COUNTRY_VOL = 30

# ════════════════════════════════════════════════════════════════════════════════
# Helper: load / reconstruct the test split for a domain
# ════════════════════════════════════════════════════════════════════════════════

def get_domain_instances_from_confidences(nli_conf: pd.DataFrame, domain: str) -> set:
    """
    Return the set of instance URIs that were evaluated for *domain*.
    These are the rows in nli_confidences.csv that have at least one
    non-NaN prediction column for the given domain prefix.
    """
    prefix = DOMAIN_PREFIX[domain] + "-n-"
    domain_cols = [c for c in nli_conf.columns
                   if c.startswith(prefix) and not c.endswith("_conf")]
    if not domain_cols:
        raise ValueError(f"No NLI prediction columns found for domain '{domain}' "
                         f"(expected prefix '{prefix}').")
    # An instance belongs to this domain's test set if at least one pred col is
    # non-NaN for it.
    mask = nli_conf[domain_cols].notna().any(axis=1)
    return set(nli_conf.loc[mask, "instance"])


def load_test_df_for_domain(domain: str, nli_conf: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the test DataFrame for a domain by:
      1. loading the enriched dataset (which has class labels + country),
      2. keeping only the instances that were actually evaluated (from
         nli_confidences.csv).
    """
    print(f"  Loading full dataset for domain '{domain}'…")
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=["instance"], keep="first")
    df["class_ids"] = df["class_ids"].apply(ast.literal_eval)
    df["classes"]   = df["classes"].apply(ast.literal_eval)

    # Merge tokenized / decomposed (needed only if we re-run rules; for NLI we
    # just need instance, country, and class columns).
    tok  = pd.read_csv(TOKENIZED_PATH).drop_duplicates(subset=["instance"])
    dec  = pd.read_csv(DECOMPOSED_PATH).drop_duplicates(subset=["instance"])
    df   = df.merge(tok[["instance", "tokenized"]], on="instance", how="left")
    df   = df.merge(dec[["instance", "decomposed"]], on="instance", how="left")

    # Keep only instances that are in the evaluated set for this domain
    evaluated = get_domain_instances_from_confidences(nli_conf, domain)
    df = df[df["instance"].isin(evaluated)].copy()

    print(f"    → {len(df)} test instances for domain '{domain}'.")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# Helper: valid countries for a domain (same logic as evaluator.py)
# ════════════════════════════════════════════════════════════════════════════════

def valid_countries_for_domain(test_df: pd.DataFrame, domain: str) -> list:
    """
    Return countries where every class has ≥ MIN_COUNTRY_VOL instances.
    Mirrors the logic in evaluator.evaluate_rules().
    """
    classes = DOMAIN_CLASSES_CORR[domain]
    valid = []
    for country in COUNTRY_DICT:
        cdf = test_df[test_df["country"] == country]
        counts = [cdf[cdf[cls] == 1].shape[0] for cls in classes]
        if counts and min(counts) >= MIN_COUNTRY_VOL:
            valid.append(country)
    return valid


# ════════════════════════════════════════════════════════════════════════════════
# Section 1: Extract rules country-F1 from experiments_final.csv
# ════════════════════════════════════════════════════════════════════════════════

def extract_rules_country_f1(exp_ids: list, col_label: str,
                              experiments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-country F1 values that are cached inside
    Parameters["country_f1"] for the given experiment IDs.

    Parameters
    ----------
    exp_ids     : one ID per domain, e.g. ["med-r-idf-4", "adm-r-idf-1", "edu-r-idf-1"]
    col_label   : name that will appear as a column in the output table
    experiments_df : loaded experiments_final.csv (with parsed Parameters)

    Returns
    -------
    DataFrame with columns ["domain", "country", col_label]
    """
    rows = []
    for exp_id in exp_ids:
        row = experiments_df[experiments_df["ID"] == exp_id]
        if row.empty:
            print(f"  WARNING: experiment '{exp_id}' not found in experiments_final.csv")
            continue
        exp = row.iloc[0]
        domain = exp["Domain"]
        params = exp["Parameters"]
        if not isinstance(params, dict):
            import ast as _ast
            try:
                params = _ast.literal_eval(params)
            except Exception:
                print(f"  WARNING: could not parse Parameters for '{exp_id}'")
                continue

        country_f1 = params.get("country_f1", {})
        if not country_f1:
            print(f"  WARNING: no country_f1 in Parameters for '{exp_id}'")
            continue

        for country, f1_val in country_f1.items():
            if f1_val is None:
                continue          # skip countries where F1 is undefined (missing class)
            rows.append({
                "domain":    domain,
                "country":   country,
                col_label:   float(f1_val),
            })
        print(f"  Extracted {len(country_f1)} country entries for '{exp_id}' ({domain}).")

    if not rows:
        return pd.DataFrame(columns=["domain", "country", col_label])

    result = pd.DataFrame(rows)
    # There should be exactly one entry per (domain, country) from these experiments.
    result = result.drop_duplicates(subset=["domain", "country"]).reset_index(drop=True)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Section 2: Compute NLI country-F1 from nli_confidences.csv
# ════════════════════════════════════════════════════════════════════════════════

def compute_nli_country_f1(nli_exp_ids: list, col_label: str,
                            nli_conf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-country F1 for NLI experiments whose per-instance binary
    predictions live in nli_confidences.csv.

    Parameters
    ----------
    nli_exp_ids  : one ID per domain, e.g. ["med-n-0-2", "adm-n-0-2", "edu-n-0-2"]
    col_label    : column name in output table
    nli_conf     : nli_confidences.csv loaded as DataFrame

    Returns
    -------
    DataFrame with columns ["domain", "country", col_label]
    """
    rows = []
    for exp_id in nli_exp_ids:
        # Infer domain from the ID prefix (e.g. "med" → "medical")
        prefix = exp_id.split("-")[0]
        domain = PREFIX_DOMAIN.get(prefix)
        if domain is None:
            print(f"  WARNING: cannot infer domain from exp_id '{exp_id}'. Skipping.")
            continue

        classes = DOMAIN_CLASSES_CORR[domain]
        pred_cols = [f"{exp_id}_{cls}" for cls in classes]

        # Check columns exist
        missing = [c for c in pred_cols if c not in nli_conf.columns]
        if missing:
            print(f"  WARNING: columns missing in nli_confidences for '{exp_id}': {missing}")
            continue

        # Restrict to instances that actually have predictions for this experiment
        domain_mask = nli_conf[pred_cols[0]].notna()
        exp_df = nli_conf[domain_mask].copy()

        if "country" not in exp_df.columns:
            print(f"  WARNING: 'country' column missing in nli_confidences.csv for '{exp_id}'")
            continue

        # Cast predictions to int (they are stored as floats)
        for col in pred_cols:
            exp_df[col] = exp_df[col].fillna(0).astype(int)

        # Per-country F1
        for country in exp_df["country"].unique():
            cdf = exp_df[exp_df["country"] == country]
            y_true = cdf[classes].values
            y_pred = cdf[pred_cols].values

            # Skip if any class is entirely absent (undefined F1)
            if (y_true.sum(axis=0) == 0).any():
                continue

            f1_val = f1_score(y_true, y_pred, average="macro", zero_division=0)
            rows.append({
                "domain":  domain,
                "country": country,
                col_label: float(f1_val),
            })

        n_countries = len(set(r["country"] for r in rows if r.get("domain") == domain))
        print(f"  Computed NLI F1 for '{exp_id}' ({domain}): {n_countries} countries.")

    if not rows:
        return pd.DataFrame(columns=["domain", "country", col_label])

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["domain", "country"]).reset_index(drop=True)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Section 3: Compute embedding country metric from correctness tables
# ════════════════════════════════════════════════════════════════════════════════

def compute_embedding_country_metric_from_correctness(
    exp_ids: list, col_label: str,
    correctness_csv: str,
    dataset_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-country mean exact-match correctness from a pre-built
    correctness table for embedding experiments.

    The correctness tables store binary per-instance exact-match correctness
    (1 = ALL class labels correctly predicted, 0 = at least one wrong).
    Since the actual per-class prediction vectors are not stored, the best
    metric we can derive without re-running inference is mean exact-match
    correctness per country (= subset accuracy).

    Parameters
    ----------
    exp_ids        : one experiment ID per domain,
                     e.g. ["med-e-similarity-0", "adm-e-similarity-0", "edu-e-similarity-0"]
    col_label      : name that will appear as a column in the output table
    correctness_csv: path to the correctness CSV (instance × experiment)
    dataset_df     : DataFrame with columns ["instance", "country"];
                     instances must be lowercase-normalised.

    Returns
    -------
    DataFrame with columns ["domain", "country", col_label]
    """
    print(f"  Loading correctness table: {correctness_csv}")
    ct = pd.read_csv(correctness_csv)

    # Handle instance column (may be named 'instance' or may be an unnamed index)
    if "instance" not in ct.columns:
        first_col = ct.columns[0]
        if first_col.startswith("Unnamed"):
            ct = ct.rename(columns={first_col: "instance"})
        else:
            # Instance might be in the index after read_csv(index_col=0)
            ct = ct.reset_index()
            ct = ct.rename(columns={ct.columns[0]: "instance"})

    # Normalise instance URIs to lowercase for matching
    ct["instance"] = ct["instance"].str.lower()

    # Pre-load all class columns from the dataset for the MIN_COUNTRY_VOL
    # filter (loaded once, reused across all exp_ids in this call).
    all_classes = sorted(set(
        cls for d in DOMAIN_CLASSES_CORR.values() for cls in d
    ))
    class_df = pd.read_csv(DATASET_PATH, usecols=["instance"] + all_classes)
    class_df = class_df.drop_duplicates(subset=["instance"], keep="first")
    class_df["instance"] = class_df["instance"].str.lower()

    rows = []
    for exp_id in exp_ids:
        if exp_id not in ct.columns:
            print(f"  WARNING: column '{exp_id}' not found in {correctness_csv} – skipping.")
            continue

        # Infer domain from the ID prefix (e.g. "med" → "medical")
        prefix = exp_id.split("-")[0]
        domain = PREFIX_DOMAIN.get(prefix)
        if domain is None:
            print(f"  WARNING: cannot infer domain from '{exp_id}' – skipping.")
            continue

        classes = DOMAIN_CLASSES_CORR[domain]

        # Merge correctness with dataset to get country
        merged = ct[["instance", exp_id]].merge(
            dataset_df[["instance", "country"]],
            on="instance", how="inner",
        )
        merged = merged.dropna(subset=[exp_id])
        merged[exp_id] = merged[exp_id].astype(int)

        # Add class columns for the MIN_COUNTRY_VOL filter
        merged = merged.merge(class_df[["instance"] + classes], on="instance", how="left")

        # Per-country metric
        for country in sorted(merged["country"].unique()):
            cdf = merged[merged["country"] == country]

            # Filter: each class must have ≥ MIN_COUNTRY_VOL positive instances
            counts = [cdf[cdf[cls] == 1].shape[0] for cls in classes]
            if not counts or min(counts) < MIN_COUNTRY_VOL:
                continue

            metric_val = cdf[exp_id].mean()
            rows.append({
                "domain":  domain,
                "country": country,
                col_label: float(metric_val),
            })

        n_countries = len([r for r in rows if r.get("domain") == domain])
        print(f"  '{exp_id}' ({domain}): {n_countries} valid countries.")

    if not rows:
        return pd.DataFrame(columns=["domain", "country", col_label])

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["domain", "country"]).reset_index(drop=True)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main: define experiment groups and build/merge the output table
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Building CD Diagram data…")
    print("=" * 65)

    # ── load experiments_final.csv ───────────────────────────────────────
    print("\nLoading experiments_final.csv…")
    exp_df = load_experiments(EXPERIMENTS_FINAL_PATH)
    print(f"  {len(exp_df)} experiments loaded.")

    # ── load nli_confidences.csv (lazy; only if needed) ──────────────────
    nli_conf = None  # loaded on first NLI block

    # ── initialise or reload existing output table ────────────────────────
    if os.path.exists(OUTPUT_PATH):
        cd_data = pd.read_csv(OUTPUT_PATH)
        print(f"\nReloaded existing output table: {OUTPUT_PATH}  ({len(cd_data)} rows)")
    else:
        cd_data = pd.DataFrame(columns=["domain", "country"])
        print("\nCreating new output table.")

    # ════════════════════════════════════════════════════════════════════
    # BLOCK 1 – Rules IDF (best):  med-r-idf-4 / adm-r-idf-1 / edu-r-idf-1
    # ════════════════════════════════════════════════════════════════════
    col_rules_idf = "Rules (IDF)"
    if col_rules_idf not in cd_data.columns:
        print(f"\n[Block 1] Extracting '{col_rules_idf}'…")
        df_block = extract_rules_country_f1(
            exp_ids=["med-r-idf-4", "adm-r-idf-1", "edu-r-idf-1"],
            col_label=col_rules_idf,
            experiments_df=exp_df,
        )
        cd_data = merge_block(cd_data, df_block, col_rules_idf)
    else:
        print(f"\n[Block 1] '{col_rules_idf}' already present – skipping.")

    # ════════════════════════════════════════════════════════════════════
    # BLOCK 2 – Rules LLM-generated:  med-r-llm-0 / adm-r-llm-0 / edu-r-llm-0
    # ════════════════════════════════════════════════════════════════════
    col_rules_llm = "Rules (LLM)"
    if col_rules_llm not in cd_data.columns:
        print(f"\n[Block 2] Extracting '{col_rules_llm}'…")
        df_block = extract_rules_country_f1(
            exp_ids=["med-r-llm-0", "adm-r-llm-0", "edu-r-llm-0"],
            col_label=col_rules_llm,
            experiments_df=exp_df,
        )
        cd_data = merge_block(cd_data, df_block, col_rules_llm)
    else:
        print(f"\n[Block 2] '{col_rules_llm}' already present – skipping.")

    # ════════════════════════════════════════════════════════════════════
    # BLOCK 3 – NLI mDeBERTa (0-shot):  med-n-0-2 / adm-n-0-2 / edu-n-0-2
    # ════════════════════════════════════════════════════════════════════
    col_nli_mdeberta = "NLI (mDeBERTa)"
    if col_nli_mdeberta not in cd_data.columns:
        print(f"\n[Block 3] Computing '{col_nli_mdeberta}' from nli_confidences…")
        if nli_conf is None:
            print("  Loading nli_confidences.csv (this may take a moment)…")
            nli_conf = pd.read_csv(NLI_CONFIDENCES_PATH, low_memory=False)
            print(f"  Loaded {len(nli_conf)} rows, {len(nli_conf.columns)} columns.")
        df_block = compute_nli_country_f1(
            nli_exp_ids=["med-n-0-2", "adm-n-0-2", "edu-n-0-2"],
            col_label=col_nli_mdeberta,
            nli_conf=nli_conf,
        )
        cd_data = merge_block(cd_data, df_block, col_nli_mdeberta)
    else:
        print(f"\n[Block 3] '{col_nli_mdeberta}' already present – skipping.")

    # ════════════════════════════════════════════════════════════════════
    # EMBEDDING BLOCKS  (use pre-built correctness tables)
    #
    # Lazy-load dataset once for all embedding blocks (need country info).
    # ════════════════════════════════════════════════════════════════════
    dataset_df = None   # loaded on first embedding block

    def _ensure_dataset():
        nonlocal dataset_df
        if dataset_df is None:
            print("  Loading dataset for country information…")
            dataset_df = pd.read_csv(DATASET_PATH, usecols=["instance", "country"])
            dataset_df = dataset_df.drop_duplicates(subset=["instance"], keep="first")
            # Normalise to lowercase (correctness tables use lowercase URIs)
            dataset_df["instance"] = dataset_df["instance"].str.lower()
            print(f"    → {len(dataset_df)} unique instances.")
        return dataset_df

    # ════════════════════════════════════════════════════════════════════
    # BLOCK 4 – ME5 Similarity:  xxx-e-similarity-0
    # ════════════════════════════════════════════════════════════════════
    col_me5_sim = "ME5 (Sim)"
    if col_me5_sim not in cd_data.columns:
        print(f"\n[Block 4] Computing '{col_me5_sim}' from correctness table…")
        df_block = compute_embedding_country_metric_from_correctness(
            exp_ids=["med-e-similarity-0", "adm-e-similarity-0", "edu-e-similarity-0"],
            col_label=col_me5_sim,
            correctness_csv=os.path.join(CORRECTNESS_DIR, "multilingual-e5_sim.csv"),
            dataset_df=_ensure_dataset(),
        )
        cd_data = merge_block(cd_data, df_block, col_me5_sim)
    else:
        print(f"\n[Block 4] '{col_me5_sim}' already present – skipping.")

    # ════════════════════════════════════════════════════════════════════
    # BLOCK 5 – ME5 SVM:  xxx-e-classifier-1
    # ════════════════════════════════════════════════════════════════════
    col_me5_svm = "ME5 (SVM)"
    if col_me5_svm not in cd_data.columns:
        print(f"\n[Block 5] Computing '{col_me5_svm}' from correctness table…")
        df_block = compute_embedding_country_metric_from_correctness(
            exp_ids=["med-e-classifier-1", "adm-e-classifier-1", "edu-e-classifier-1"],
            col_label=col_me5_svm,
            correctness_csv=os.path.join(CORRECTNESS_DIR, "multilingual-e5_clf.csv"),
            dataset_df=_ensure_dataset(),
        )
        cd_data = merge_block(cd_data, df_block, col_me5_svm)
    else:
        print(f"\n[Block 5] '{col_me5_svm}' already present – skipping.")

    # ════════════════════════════════════════════════════════════════════
    # BLOCK 6 – Finetuned ME5 Similarity:  xxx-e-similarity-12
    # ════════════════════════════════════════════════════════════════════
    col_ft_sim = "FT-ME5 (Sim)"
    if col_ft_sim not in cd_data.columns:
        print(f"\n[Block 6] Computing '{col_ft_sim}' from correctness table…")
        df_block = compute_embedding_country_metric_from_correctness(
            exp_ids=["med-e-similarity-12", "adm-e-similarity-12", "edu-e-similarity-12"],
            col_label=col_ft_sim,
            correctness_csv=os.path.join(CORRECTNESS_DIR, "finetuned-me5_sim.csv"),
            dataset_df=_ensure_dataset(),
        )
        cd_data = merge_block(cd_data, df_block, col_ft_sim)
    else:
        print(f"\n[Block 6] '{col_ft_sim}' already present – skipping.")

    # ════════════════════════════════════════════════════════════════════
    # BLOCK 7 – Finetuned ME5 SVM:  xxx-e-classifier-9
    # ════════════════════════════════════════════════════════════════════
    col_ft_svm = "FT-ME5 (SVM)"
    if col_ft_svm not in cd_data.columns:
        print(f"\n[Block 7] Computing '{col_ft_svm}' from correctness table…")
        df_block = compute_embedding_country_metric_from_correctness(
            exp_ids=["med-e-classifier-9", "adm-e-classifier-9", "edu-e-classifier-9"],
            col_label=col_ft_svm,
            correctness_csv=os.path.join(CORRECTNESS_DIR, "finetuned-me5_clf.csv"),
            dataset_df=_ensure_dataset(),
        )
        cd_data = merge_block(cd_data, df_block, col_ft_svm)
    else:
        print(f"\n[Block 7] '{col_ft_svm}' already present – skipping.")

    # ════════════════════════════════════════════════════════════════════
    # Future blocks can be added here in the same pattern.
    # ════════════════════════════════════════════════════════════════════

    # ── save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cd_data.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved output table to: {OUTPUT_PATH}")
    print(f"  Shape: {cd_data.shape}  (rows × cols)")
    print(f"  Columns: {list(cd_data.columns)}")
    print(f"  Domains: {cd_data['domain'].unique().tolist() if 'domain' in cd_data.columns else 'n/a'}")
    if "country" in cd_data.columns:
        print(f"  Countries per domain:")
        for dom, grp in cd_data.groupby("domain"):
            print(f"    {dom}: {len(grp)} countries")

    # Quick peek
    print("\nFirst 10 rows:")
    print(cd_data.head(10).to_string(index=False))
    return cd_data


# ════════════════════════════════════════════════════════════════════════════════
# Utility: merge a new block into the master table
# ════════════════════════════════════════════════════════════════════════════════

def merge_block(master: pd.DataFrame, block: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Left-join *block* (which has columns ["domain", "country", col]) into
    *master* on (domain, country), creating the column if missing.
    Existing rows in master that don't appear in block get NaN for *col*.
    """
    if block.empty:
        print(f"  WARNING: block for '{col}' is empty – column not added.")
        return master

    if master.empty or len(master.columns) == 0 or "domain" not in master.columns:
        # First block ever: initialise with (domain, country) index from block
        return block[["domain", "country", col]].copy()

    # Merge on (domain, country); outer so we keep all countries from both sides
    merged = master.merge(
        block[["domain", "country", col]],
        on=["domain", "country"],
        how="outer",
    )
    n_new = block.shape[0]
    n_matched = merged[col].notna().sum()
    print(f"  Merged '{col}': {n_new} block rows → {n_matched} matched in master.")
    return merged


if __name__ == "__main__":
    main()
