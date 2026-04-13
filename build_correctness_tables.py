"""
build_correctness_tables.py
============================
Standalone script to build and cache per-experiment correctness tables for
the stratified permutation test pipeline.

Run from the project root:
    python build_correctness_tables.py

What it does:
  - Loads experiments.csv and the master dataset via load_dataset()
  - For each embedding model, checks which correctness tables already exist
    in results/correctness_tables/
  - Only computes missing experiment IDs (incremental)
  - Saves each experiment result immediately to disk so interruptions are safe
  - Logs every step with full detail: paths accessed, skips, errors

Log output is written to: results/correctness_tables/build.log
"""

import os
import re
import sys
import ast
import logging
import traceback
from itertools import combinations

import numpy as np
import pandas as pd

# ── Setup project root ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from orgpackage.aux import load_experiments, load_trained_model, load_dataset
from orgpackage.config import DOMAIN_CLASSES_CORR

# ── Constants ─────────────────────────────────────────────────────────────────
CORRECTNESS_DIR = os.path.join(PROJECT_ROOT, "results", "correctness_tables")
EMBEDDINGS_DIR  = os.path.join(PROJECT_ROOT, "results", "embeddings")
DOMAINS         = ["medical", "administrative", "education"]
ALL_LABELS      = ["hospital", "university_hospital", "local_government",
                   "primary_school", "secondary_school"]

os.makedirs(CORRECTNESS_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(CORRECTNESS_DIR, "build.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_uri(uri: str) -> str:
    if not isinstance(uri, str):
        return uri
    return uri.strip().lower().replace("https://", "http://")


def normalize_path(raw: str) -> str:
    """
    Converts the escaped paths stored in experiments.csv (e.g. '.\\/results\\/...')
    to a usable relative path and verifies it exists.
    """
    if not isinstance(raw, str):
        return raw
    p = re.sub(r"\\+/", "/", raw)   # \\/ or \\\\/ → /
    p = p.replace("\\", "/")         # any stray backslashes
    p = re.sub(r"^\./", "", p)       # strip leading ./
    if os.path.exists(p):
        return p
    # Fallback: look just in trained_models by filename
    fallback = os.path.join("results", "trained_models", os.path.basename(p))
    if os.path.exists(fallback):
        log.debug("  Path fallback: %s → %s", raw, fallback)
        return fallback
    return p   # return non-existent path so caller can log the miss


def load_embeddings(model_name: str, dataset: pd.DataFrame):
    """
    Load and parse an embedding CSV, merge with master dataset.
    Returns (merged_df, embedding_col_name) or (empty_df, col_name) on failure.
    """
    path = os.path.join(EMBEDDINGS_DIR, f"{model_name}_embeddings.csv")
    col  = f"{model_name}_embedding"

    if not os.path.exists(path):
        log.warning("  [SKIP] Embedding file not found: %s", path)
        return pd.DataFrame(), col

    log.info("  Loading embeddings from %s", path)
    df = pd.read_csv(path, low_memory=False)

    if col not in df.columns:
        log.warning("  [SKIP] Column '%s' missing in %s (columns: %s)",
                    col, path, list(df.columns))
        return pd.DataFrame(), col

    def _parse(v):
        if pd.isna(v):
            return None
        try:
            return np.array(ast.literal_eval(
                str(v).replace("np.array(", "[").replace(")", "]")
            ))
        except Exception:
            return None

    df[col] = df[col].apply(_parse)

    def _is_valid(v):
        return (isinstance(v, np.ndarray)
                and np.issubdtype(v.dtype, np.number)
                and not np.isnan(v).any())

    before = len(df)
    df = df[df[col].apply(_is_valid)].copy()
    log.info("  %d / %d rows survived embedding parse", len(df), before)

    if df.empty:
        log.warning("  [SKIP] No valid embeddings after parse for %s", model_name)
        return pd.DataFrame(), col

    df["instance"] = df["instance"].apply(normalize_uri)
    df = df[["instance", col]]

    merged = df.merge(dataset[["instance"] + ALL_LABELS], on="instance", how="inner")
    log.info("  %d instances after merge with master dataset", len(merged))

    if merged.empty:
        log.warning("  [SKIP] Merge produced 0 rows for %s — check URI formats", model_name)

    return merged, col


# ── Similarity correctness ────────────────────────────────────────────────────

def _compute_sim_one(eid: str, params: dict, domain: str,
                     df: pd.DataFrame, col: str,
                     X_n: np.ndarray) -> pd.Series | None:
    """Compute per-instance correctness for one similarity experiment."""
    cl        = DOMAIN_CLASSES_CORR[domain]
    threshold = 1.0 - params["distance"]
    multiclass = params.get("structure") == "2-multiclass"
    X_dim     = X_n.shape[1]

    all_sims = np.full((len(X_n), len(cl)), -1.0)

    for i, cls_name in enumerate(cl):
        protos_raw = params["prototypes"].get(cls_name)
        if protos_raw is None:
            log.debug("    Class '%s' has no prototypes in %s — skipping", cls_name, eid)
            continue

        plist = list(protos_raw.values()) if isinstance(protos_raw, dict) else [protos_raw]
        p_vecs = []
        for v in plist:
            if v is None:
                continue
            v_arr = np.array(v).flatten()
            if not np.issubdtype(v_arr.dtype, np.number):
                log.debug("    Prototype for class '%s' in %s is non-numeric, skipping", cls_name, eid)
                continue
            if np.isnan(v_arr).any():
                log.debug("    Prototype for class '%s' in %s contains NaN, skipping", cls_name, eid)
                continue
            if v_arr.shape[0] != X_dim:
                log.warning("    Prototype dim mismatch for class '%s' in %s: "
                            "got %d, expected %d — skipping vector",
                            cls_name, eid, v_arr.shape[0], X_dim)
                continue
            p_vecs.append(v_arr)

        if not p_vecs:
            log.warning("    No valid prototype vectors for class '%s' in %s", cls_name, eid)
            continue

        Pn    = np.stack(p_vecs)
        norms = np.linalg.norm(Pn, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Pn /= norms
        all_sims[:, i] = (X_n @ Pn.T).max(axis=1)

    yp = np.zeros((len(X_n), len(cl)), dtype=int)
    if multiclass:
        yp[all_sims >= threshold] = 1
    else:
        best = np.argmax(all_sims, axis=1)
        mx   = all_sims.max(axis=1)
        for ri, ci in enumerate(best):
            if mx[ri] >= threshold:
                yp[ri, ci] = 1

    hits = (yp == df[cl].values).all(axis=1).astype(int)
    return pd.Series(hits, index=df["instance"], name=eid)


# ── Classifier correctness ────────────────────────────────────────────────────

def _compute_clf_one(eid: str, params: dict, domain: str,
                     df: pd.DataFrame, col: str,
                     X: np.ndarray) -> pd.Series | None:
    """Compute per-instance correctness for one classifier experiment."""
    cl = DOMAIN_CLASSES_CORR[domain]
    st = params.get("structure")

    if st == "nested-class":
        raw_paths = {
            "hospital":            params.get("trained_classifier_hospital"),
            "university_hospital": params.get("trained_classifier_university_hospital"),
        }
    else:
        raw_paths = {"main": params.get("trained_classifier")}

    # Resolve and validate paths
    resolved = {}
    for key, raw in raw_paths.items():
        if raw is None:
            log.warning("    [SKIP] %s: key '%s' missing from Parameters", eid, key)
            return None
        norm = normalize_path(raw)
        if not os.path.exists(norm):
            log.warning("    [SKIP] %s: pickle not found.\n"
                        "             raw path in params : %s\n"
                        "             resolved path tried: %s",
                        eid, raw, norm)
            return None
        resolved[key] = norm
        log.debug("    Resolved pickle for '%s': %s", key, norm)

    try:
        if st == "nested-class":
            clf_h = load_trained_model(resolved["hospital"])
            clf_u = load_trained_model(resolved["university_hospital"])
            yh    = clf_h.predict(X)
            yu    = np.zeros_like(yh)
            pos   = np.where(yh == 1)[0]
            if len(pos):
                yu[pos] = clf_u.predict(X[pos])
            hits = ((yh == df["hospital"].values) &
                    (yu == df["university_hospital"].values)).astype(int)
        else:
            clf = load_trained_model(resolved["main"])
            raw_pred = clf.predict(X)
            yt = df[cl].values
            if yt.shape[1] > 1:
                yp = np.zeros_like(yt)
                if raw_pred.ndim == 1:
                    for i_r, v in enumerate(raw_pred):
                        if v < yt.shape[1]:
                            yp[i_r, v] = 1
                else:
                    yp = raw_pred.astype(int)
            else:
                yp = raw_pred.reshape(-1, 1).astype(int)
            hits = (yp == yt).all(axis=1).astype(int)

        return pd.Series(hits, index=df["instance"], name=eid)

    except Exception:
        log.error("    [ERROR] %s during prediction:\n%s", eid, traceback.format_exc())
        return None


# ── Incremental cache manager ─────────────────────────────────────────────────

def build_table(model_name: str, method_type: str,
                requested_ids: list, exps_df: pd.DataFrame,
                dataset: pd.DataFrame) -> pd.DataFrame:
    """
    For a given model + method_type ('sim' or 'clf'), compute only the
    missing experiment IDs and append to the CSV cache immediately.
    Returns a DataFrame with all successfully computed columns.
    """
    cache_path = os.path.join(CORRECTNESS_DIR, f"{model_name}_{method_type}.csv")

    # Load existing cache
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path, index_col=0)
        cache_df.index = cache_df.index.map(normalize_uri)
        log.info("  Loaded cache: %s  (%d columns)", cache_path, len(cache_df.columns))
    else:
        cache_df = pd.DataFrame()
        log.info("  No cache found at %s — will create fresh", cache_path)

    cached_ids    = set(cache_df.columns) if not cache_df.empty else set()
    missing_ids   = [eid for eid in requested_ids if eid not in cached_ids]
    available_ids = [eid for eid in requested_ids if eid in cached_ids]

    log.info("  [%s %s]  %d already cached | %d to compute",
             model_name, method_type, len(available_ids), len(missing_ids))

    if not missing_ids:
        return cache_df[available_ids]

    # Load embeddings once
    df, col = load_embeddings(model_name, dataset)
    if df.empty:
        return cache_df[available_ids] if available_ids else pd.DataFrame()

    # Pre-compute shared matrices
    X = np.stack(df[col].values)
    if method_type == "sim":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        X_n = X / norm

    # Compute experiment by experiment, save immediately
    for eid in missing_ids:
        log.info("    Computing %s ...", eid)
        row_matches = exps_df[exps_df["ID"] == eid]
        if row_matches.empty:
            log.warning("    [SKIP] %s not found in experiments DataFrame", eid)
            continue

        row    = row_matches.iloc[0]
        params = row["Parameters"]
        domain = row["Domain"]

        try:
            if method_type == "sim":
                result = _compute_sim_one(eid, params, domain, df, col, X_n)
            else:
                result = _compute_clf_one(eid, params, domain, df, col, X)
        except Exception:
            log.error("    [ERROR] Unhandled exception for %s:\n%s",
                      eid, traceback.format_exc())
            result = None

        if result is not None:
            acc = result.mean()
            log.info("    %s  →  accuracy=%.4f  (%d instances)", eid, acc, len(result))
            new_col = result.to_frame()
            cache_df = new_col if cache_df.empty else \
                       cache_df.join(new_col, how="outer").fillna(0).astype(int)
            cache_df.to_csv(cache_path)
            log.info("    Saved to %s", cache_path)
        else:
            log.info("    %s → skipped (see warnings above)", eid)

    all_available = [eid for eid in requested_ids if eid in cache_df.columns]
    return cache_df[all_available] if all_available else pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("build_correctness_tables.py  —  start")
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Output dir   : %s", CORRECTNESS_DIR)
    log.info("=" * 70)

    # 1. Load experiments
    log.info("Loading experiments…")
    exps = load_experiments()
    emb_exps = exps[exps["Technique"] == "embedding"]
    sim_exps  = emb_exps[emb_exps["Method"] == "similarity"]
    clf_exps  = emb_exps[emb_exps["Method"] == "classifier"]
    log.info("  %d similarity | %d classifier experiments", len(sim_exps), len(clf_exps))

    # 2. Load master dataset
    log.info("Loading master dataset…")
    dataset = load_dataset()
    dataset["instance"] = (dataset["instance"]
                           .str.strip().str.lower()
                           .str.replace("https://", "http://"))
    for c in ALL_LABELS:
        if c not in dataset.columns:
            dataset[c] = 0
        else:
            dataset[c] = dataset[c].fillna(0).astype(int)
    log.info("  %d instances loaded", len(dataset))

    # 3. Build index: model → {n_shot: [ids]} for similarity
    sim_index: dict = {}
    for _, r in sim_exps.iterrows():
        p = r["Parameters"]
        if isinstance(p, dict):
            sim_index.setdefault(p.get("model"), {}) \
                     .setdefault(p.get("n_shot"), []) \
                     .append(r["ID"])

    # 4. Build index: model → {head_type: [ids]} for classifiers
    clf_index: dict = {}
    for _, r in clf_exps.iterrows():
        p = r["Parameters"]
        if isinstance(p, dict):
            t = "lr" if "solver" in p else ("svm" if "kernel" in p else "unknown")
            clf_index.setdefault(p.get("model"), {}) \
                     .setdefault(t, []) \
                     .append(r["ID"])

    log.info("Similarity models : %s", list(sim_index.keys()))
    log.info("Classifier models : %s", list(clf_index.keys()))

    # 5. Build similarity tables
    log.info("")
    log.info("── SIMILARITY TABLES ─────────────────────────────────────────────")
    sim_ct: dict = {}
    for m in sim_index:
        if not m:
            continue
        ids = [i for s in sim_index[m].values() for i in s]
        log.info("Model: %s  (%d experiments)", m, len(ids))
        ct = build_table(m, "sim", ids, sim_exps, dataset)
        if not ct.empty:
            sim_ct[m] = ct

    # 6. Build classifier tables
    log.info("")
    log.info("── CLASSIFIER TABLES ─────────────────────────────────────────────")
    clf_ct: dict = {}
    for m in clf_index:
        if not m:
            continue
        ids = [i for t in clf_index[m].values() for i in t]
        log.info("Model: %s  (%d experiments)", m, len(ids))
        ct = build_table(m, "clf", ids, clf_exps, dataset)
        if not ct.empty:
            clf_ct[m] = ct

    # 7. Summary
    log.info("")
    log.info("=" * 70)
    log.info("DONE")
    log.info("Similarity tables ready : %s", list(sim_ct.keys()))
    log.info("Classifier tables ready : %s", list(clf_ct.keys()))
    for m, ct in {**sim_ct, **clf_ct}.items():
        log.info("  %-25s  %d experiments  x  %d instances",
                 m, len(ct.columns), len(ct))
    log.info("Full log written to: %s", LOG_PATH)


if __name__ == "__main__":
    main()
