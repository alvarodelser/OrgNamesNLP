"""
build_embedding_predictions.py
===============================
Generate per-instance prediction tables for all *embedding* experiments
(cosine-similarity and classifier-head), one CSV per domain.

Output format (same as the rule-based correctness table)
---------------------------------------------------------
  rows    → entity instances
  columns → {exp_id}_{class}   with values 0 / 1

e.g.  med-e-similarity-0_hospital | med-e-similarity-0_university_hospital | ...

Saved to:  results/embedding_predictions_{domain}.csv

Usage
-----
    python build_embedding_predictions.py [--domain medical administrative education]
                                          [--technique similarity classifier all]
                                          [--experiments results/experiments.csv]
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from orgpackage.aux import load_dataset, load_experiments, load_embeddings, load_trained_model, prepare_labels
from orgpackage.config import DOMAIN_CLASSES_CORR, STRUCTURE_MAPPING
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDINGS_BASE = "./results/embeddings"
OUTPUT_DIR      = "./results"


def _safe_array(x):
    """Convert a stored embedding value to a 2D numpy array, or None."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        try:
            x = np.array(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return None
    if isinstance(x, np.ndarray):
        if np.isnan(x).any():
            return None
        return x.reshape(1, -1) if x.ndim == 1 else x
    return None


def _load_instance_embeddings(model: str) -> pd.DataFrame:
    path = os.path.join(EMBEDDINGS_BASE, f"{model}_embeddings.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return load_embeddings(path)


def _load_label_embeddings() -> pd.DataFrame:
    path = os.path.join(EMBEDDINGS_BASE, "label_embeddings.csv")
    return load_embeddings(path)


# ──────────────────────────────────────────────────────────────────────────────
# Similarity prediction (replicates evaluate_similarity_experiment logic)
# ──────────────────────────────────────────────────────────────────────────────

def _predict_similarity(df: pd.DataFrame, exp: pd.Series) -> dict[str, np.ndarray]:
    """
    Return {class → binary 0/1 array aligned to df.index} for a
    cosine-similarity experiment.
    """
    params        = exp["Parameters"]
    classes       = DOMAIN_CLASSES_CORR[exp["Domain"]]
    preds = {cls: np.zeros(len(df), dtype=int) for cls in classes}

    if not isinstance(params, dict):
        print(f"  WARNING: Parameters for {exp['ID']} is not a valid dictionary (possibly truncated in CSV). Skipping.")
        return preds

    model         = params["model"]
    embedding_col = f"{model}_embedding"
    distance      = params["distance"]
    threshold     = 1.0 - distance
    prototypes    = params["prototypes"]
    structure     = params["structure"]
    is_multiclass = structure == "2-multiclass"

    for row_pos, (idx, row) in enumerate(df.iterrows()):
        x = _safe_array(row.get(embedding_col))
        if x is None:
            continue

        sims = {}
        for cls in classes:
            proto = prototypes.get(cls)
            if proto is None:
                continue
            if isinstance(proto, dict):           # few-shot: country → array
                best = -1.0
                for p in proto.values():
                    p = _safe_array(p)
                    if p is None:
                        continue
                    try:
                        best = max(best, cosine_similarity(x, p)[0][0])
                    except Exception:
                        pass
                if best != -1.0:
                    sims[cls] = best
            else:                                 # 0-shot / 1-shot
                p = _safe_array(proto)
                if p is None:
                    continue
                try:
                    sims[cls] = cosine_similarity(x, p)[0][0]
                except Exception:
                    pass

        if not sims:
            continue

        if is_multiclass:
            for cls, sim in sims.items():
                if sim >= threshold:
                    preds[cls][row_pos] = 1
        else:
            valid = {k: v for k, v in sims.items() if v != -1.0}
            if valid:
                best_cls = max(valid, key=valid.get)
                if valid[best_cls] >= threshold:
                    preds[best_cls][row_pos] = 1

    return preds



# ──────────────────────────────────────────────────────────────────────────────
# Classifier prediction (loads trained model from path stored in Parameters)
# ──────────────────────────────────────────────────────────────────────────────

def _predict_classifier(df: pd.DataFrame, exp: pd.Series) -> dict[str, np.ndarray]:
    """
    Return {class → binary 0/1 array aligned to df.index} for a
    classifier-head experiment.  Loads the serialised sklearn model.
    """
    params        = exp["Parameters"]
    classes       = DOMAIN_CLASSES_CORR[exp["Domain"]]
    preds = {cls: np.zeros(len(df), dtype=int) for cls in classes}

    if not isinstance(params, dict):
        print(f"  WARNING: Parameters for {exp['ID']} is not a valid dictionary. Skipping.")
        return preds

    model         = params["model"]
    embedding_col = f"{model}_embedding"
    structure     = params.get("structure", "flat")


    # Build X, keeping track of which rows have valid embeddings
    X, valid_pos = [], []
    for pos, (_, row) in enumerate(df.iterrows()):
        arr = _safe_array(row.get(embedding_col))
        if arr is not None:
            X.append(arr.flatten())
            valid_pos.append(pos)

    preds = {cls: np.zeros(len(df), dtype=int) for cls in classes}

    if not X:
        print(f"  WARNING: No valid embeddings for {exp['ID']}")
        return preds

    X = np.array(X)

    # Load classifier
    try:
        if structure == "nested-class":
            clf = {
                name: load_trained_model(params[f"trained_classifier_{name}"])
                for name in ["hospital", "university_hospital"]
                if f"trained_classifier_{name}" in params
            }
        else:
            clf = load_trained_model(params["trained_classifier"])
    except Exception as e:
        print(f"  WARNING: Could not load classifier for {exp['ID']}: {e}")
        return preds

    # Predict
    try:
        if structure == "nested-class" and isinstance(clf, dict):
            hospital_clf = clf["hospital"]
            y_hosp = hospital_clf.predict(X)

            y_pred = np.zeros((len(X), len(classes)), dtype=int)
            hosp_idx = classes.index("hospital")
            y_pred[:, hosp_idx] = y_hosp

            if "university_hospital" in classes and "university_hospital" in clf:
                univ_clf = clf["university_hospital"]
                univ_idx = classes.index("university_hospital")
                hosp_rows = np.where(y_hosp == 1)[0]
                if len(hosp_rows) > 0:
                    y_pred[hosp_rows, univ_idx] = univ_clf.predict(X[hosp_rows])

            for ci, cls in enumerate(classes):
                sub = np.zeros(len(df), dtype=int)
                for arr_pos, df_pos in enumerate(valid_pos):
                    sub[df_pos] = y_pred[arr_pos, ci]
                preds[cls] = sub
        else:
            y_pred = clf.predict(X)

            if y_pred.ndim == 1:
                # 2-class or single output → map to the single class
                cls = classes[0]
                sub = np.zeros(len(df), dtype=int)
                for arr_pos, df_pos in enumerate(valid_pos):
                    sub[df_pos] = int(y_pred[arr_pos])
                preds[cls] = sub
            else:
                # Multi-output (e.g. 3-multiclass)
                for ci, cls in enumerate(classes):
                    sub = np.zeros(len(df), dtype=int)
                    for arr_pos, df_pos in enumerate(valid_pos):
                        sub[df_pos] = int(y_pred[arr_pos, ci])
                    preds[cls] = sub

    except Exception as e:
        print(f"  WARNING: Prediction failed for {exp['ID']}: {e}")

    return preds


# ──────────────────────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────────────────────

def build_embedding_tables(
    domain: str,
    test_df: pd.DataFrame,
    experiments_path: str = "./results/experiments.csv",
    techniques: list[str] = ("similarity", "classifier"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-instance prediction and correctness tables for all embedding
    experiments of the given domain.

    Returns
    -------
    (df_classification, df_correctness)
    """
    exps = load_experiments(experiments_path)
    domain_exps = exps[
        (exps["Domain"] == domain) &
        (exps["Technique"] == "embedding") &
        (exps["Method"].isin(techniques))
    ].copy()

    if domain_exps.empty:
        print(f"No embedding experiments found for domain={domain!r}")
        return pd.DataFrame(), pd.DataFrame()

    classes = DOMAIN_CLASSES_CORR[domain]

    # Base attributes + ground truth labels
    base_cols = ["instance", "names", "country"] + classes
    df_clf = test_df[base_cols].reset_index(drop=True).copy()
    df_corr = test_df[base_cols].reset_index(drop=True).copy()

    # Group by model for efficiency
    def _extract_model(p):
        if isinstance(p, dict):
            return p.get("model", "")
        return ""
        
    domain_exps["_model"] = domain_exps["Parameters"].apply(_extract_model)
    domain_exps = domain_exps[domain_exps["_model"] != ""]

    for model, group in domain_exps.groupby("_model"):
        print(f"\n  Loading embeddings for model: {model}")
        try:
            instance_embs = _load_instance_embeddings(model)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        embedding_col = f"{model}_embedding"
        merged = test_df.merge(
            instance_embs[["instance", embedding_col]],
            on="instance",
            how="left",
        )

        for _, exp in group.iterrows():
            method = exp["Method"]
            exp_id = exp["ID"]
            print(f"    Processing [{method}] {exp_id} …", end=" ", flush=True)

            try:
                if method == "similarity":
                    preds = _predict_similarity(merged, exp)
                elif method == "classifier":
                    preds = _predict_classifier(merged, exp)
                else:
                    continue

                # 1. Add columns to classification table
                for cls in classes:
                    df_clf[f"{exp_id}_{cls}"] = preds[cls]

                # 2. Add columns to correctness table (Exact Vector Match)
                y_pred = np.stack([preds[cls] for cls in classes], axis=1)
                y_true = df_clf[classes].values
                df_corr[f"{exp_id}_correct"] = (y_pred == y_true).all(axis=1).astype(int)

                print("OK")

            except Exception as e:
                print(f"ERROR — {e}")

    return df_clf, df_corr


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_splits(data):
    train_medgov_full, test_medgov = train_test_split(data, test_size=0.5, random_state=42)
    train_edu_full,   test_edu     = train_test_split(data, test_size=0.2, random_state=42)
    return {
        "medical":        test_medgov,
        "administrative": test_medgov,
        "education":      test_edu,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build per-instance classification and correctness tables for embedding experiments"
    )
    parser.add_argument(
        "--domain", nargs="+",
        default=["medical", "administrative", "education"],
        help="Domains to process",
    )
    parser.add_argument(
        "--technique", nargs="+",
        default=["similarity", "classifier"],
        choices=["similarity", "classifier"],
        help="Embedding method(s) to include",
    )
    parser.add_argument(
        "--experiments",
        default="./results/experiments.csv",
        help="Path to experiments.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to write output CSVs",
    )
    args = parser.parse_args()

    print("Loading dataset …", flush=True)
    data = load_dataset()
    tests = build_splits(data)

    os.makedirs(args.output_dir, exist_ok=True)

    for domain in args.domain:
        print(f"\n{'='*60}")
        print(f"  Domain: {domain.upper()}")
        print(f"{'='*60}")

        test_df = tests[domain]

        df_clf, df_corr = build_embedding_tables(
            domain      = domain,
            test_df     = test_df,
            experiments_path = args.experiments,
            techniques  = args.technique,
        )

        if df_clf.empty:
            continue

        # Save Classification Table
        clf_path = os.path.join(args.output_dir, f"embedding_classification_{domain}.csv")
        df_clf.to_csv(clf_path, index=False)
        print(f"  Classification saved → {clf_path}")

        # Save Correctness Table
        corr_path = os.path.join(args.output_dir, f"embedding_correctness_{domain}.csv")
        df_corr.to_csv(corr_path, index=False)
        print(f"  Correctness saved → {corr_path}")


if __name__ == "__main__":
    main()
