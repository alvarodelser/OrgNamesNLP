from __future__ import annotations
import re
import time
import json
import os
from typing import Optional, Union, List, Dict
import pandas as pd
from orgpackage.config import DOMAIN_CLASSES_CORR, COUNTRY_DICT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D




def plot_word_recall_per_country(
    coverage_df,
    tests,
    exp_id,
    cls,
    word,
    title_suffix="",
    annotate=True,
    figsize=(10, 5),
):
    """
    Plot per-country *recall* for a single word and class:

      - x-axis: country
      - y-axis: recall = TP / (# true positives for that class in that country)
      - black bar: recall for this word
      - red bar (optional): FP rate relative to all negatives in that country
      - annotation: the word inside/above each bar
    """
    df = coverage_df[
        (coverage_df["exp_id"] == exp_id)
        & (coverage_df["cls"] == cls)
        & (coverage_df["word"] == word)
    ].copy()

    if df.empty:
        print(f"No coverage found for word='{word}', class='{cls}', exp_id='{exp_id}'.")
        return

    domain = df["domain"].iloc[0]
    if domain not in tests:
        raise ValueError(
            f"`tests` dict with key '{domain}' not found in `tests`."
        )
    test_df = tests[domain]

    df = df.sort_values("country")
    countries = df["country"].tolist()
    recalls = []
    fp_rates = []

    for country in countries:
        df_country = test_df[test_df["country"] == country]
        if df_country.empty:
            recalls.append(0.0)
            fp_rates.append(0.0)
            continue

        df_word = df[df["country"] == country]
        if df_word.empty:
            recalls.append(0.0)
            fp_rates.append(0.0)
            continue

        tp = df_word["tp"].iloc[0]
        fp = df_word["fp"].iloc[0]

        # Number of true positive cases for this class in this country
        n_pos = (df_country[cls] == 1).sum()

        # Recall: TP / (# true positives)
        recall = tp / n_pos if n_pos > 0 else 0.0
        # FP proportion: FP / (# true positives), as requested
        fp_rate = fp / n_pos if n_pos > 0 else 0.0

        recalls.append(recall)
        fp_rates.append(fp_rate)

    x = range(len(countries))
    fig, ax = plt.subplots(figsize=figsize)

    # Black bars: recall
    bars_recall = ax.bar(
        x,
        recalls,
        color="black",
        alpha=0.8,
        label="Recall (TP / #positives)",
    )

    # Red bars: FP proportion relative to true positives, drawn just below the main bar
    # (optional visual; can be commented out if not desired)
    bars_fp = ax.bar(
        x,
        fp_rates,
        bottom=0.0,
        color="red",
        alpha=0.4,
        label="FP proportion (FP / #positives)",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(countries, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Country")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend()

    if annotate:
        for i, r in enumerate(recalls):
            if r <= 0:
                continue
            ax.text(
                i,
                r + 0.02,
                word,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                rotation=90,
            )

    plt.tight_layout()
    plt.show()

def plot_word_coverage_all_countries(
    coverage_df,
    tests,
    exp_id,
    title_suffix="",
    figsize=(7, 25),
    top_k_words=None,
    output_path: Optional[str] = None,
):
    """
    Small multiples version of the coverage plot.

    - Rows: countries
    - Columns: classes
    - In each subplot (country, class):
        * One horizontal black bar per word
        * Red horizontal segment near the baseline for false positives
        * X-axis: 0 .. n, where n is the number of true organizations
          (true positives for that class in that country)
        * Vertical dashed grid lines at percentage positions (annotated)
    - Y-axis ticks of each subplot are the words.
    - Row labels (countries) appear vertically to the left of the leftmost plots.
    - Column labels (classes) are titles on the top row.
    - output_path: If set, save the figure here (PNG/PDF).
    """
    df = coverage_df[coverage_df["exp_id"] == exp_id].copy()
    if df.empty:
        print(f"No coverage found for exp_id='{exp_id}'.")
        return

    domain = df["domain"].iloc[0]
    domain_classes = DOMAIN_CLASSES_CORR[domain]
    # Use all classes for columns (e.g. hospital, university_hospital, etc.)
    classes = domain_classes

    test_df = tests[domain]

    countries = sorted(df["country"].unique())
    n_cols = len(classes)

    # Split countries into two halves
    mid = (len(countries) + 1) // 2
    halves = [
        ("top", countries[:mid]),
        ("bottom", countries[mid:])
    ]

    for suffix, sub_countries in halves:
        if not sub_countries:
            continue
            
        n_rows = len(sub_countries)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0], figsize[1] / 2), sharex=False, sharey=False
        )

        # Normalize axes to 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for row_idx, country in enumerate(sub_countries):
            df_country_test = test_df[test_df["country"] == country]
            df_c = df[df["country"] == country]
            country_meta = COUNTRY_DICT.get(country, {})
            country_name = country_meta.get("country", country)

            for col_idx, cls in enumerate(classes):
                ax = axes[row_idx, col_idx]

                df_cls = df_c[df_c["cls"] == cls].copy()
                df_country_cls = df_country_test[df_country_test[cls] == 1] if not df_country_test.empty else pd.DataFrame()
                denom_pos = float(len(df_country_cls))

                if df_country_test.empty or denom_pos <= 0 or df_cls.empty:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8, color="gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
                    if row_idx == 0: ax.set_title(cls, fontsize=11)
                    continue

                df_cls = df_cls.sort_values("total", ascending=False)
                if top_k_words is not None: df_cls = df_cls.head(top_k_words)
                if df_cls.empty:
                    ax.axis("off")
                    continue

                y_positions, word_labels = [], []
                max_x = denom_pos
                y_scale = 1.0
                n_words = len(df_cls)
                y_range = max(1.0, n_words * y_scale)
                fp_offset, fp_vert = 0.03 * y_range, 0.1 * y_range

                for y_idx, (_, row) in enumerate(df_cls.iterrows()):
                    y = y_idx * y_scale
                    tp_count, fp_count = row["tp"], row["fp"]
                    if tp_count == 0 and fp_count == 0: continue
                    ax.hlines(y, 0.0, tp_count, colors="black", linewidth=1.2)
                    if tp_count > 0: ax.vlines(tp_count, y, y + fp_vert, colors="black", linewidth=1.0)
                    if fp_count > 0:
                        y_red = y - fp_offset
                        if fp_count <= max_x:
                            ax.hlines(y_red, 0.0, fp_count, colors="red", linewidth=1.2)
                            ax.vlines(fp_count, y_red, y_red - fp_vert, colors="red", linewidth=1)
                        else:
                            ax.hlines(y_red, 0.0, max_x, colors="red", linewidth=1.2)
                            ax.annotate("", xy=(max_x, y_red), xytext=(max_x - 0.02 * max_x, y_red),
                                        arrowprops=dict(arrowstyle="->", color="red", linewidth=1.0))
                            ax.text(max_x, y_red + 0.6 * fp_vert, str(int(fp_count)),
                                    ha="right", va="bottom", fontsize=6, color="red")
                    y_positions.append(y)
                    word_labels.append(row["word"])

                if not y_positions:
                    ax.axis("off")
                    continue

                ax.set_ylim(min(y_positions) - 0.2 * y_range, max(y_positions) + 0.2 * y_range)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(word_labels, fontsize=7)
                ax.set_xlim(0, max_x)
                perc_fracs = np.linspace(0.0, 1.0, 6)
                for i, frac in enumerate(perc_fracs):
                    x_pos = frac * max_x
                    ax.axvline(x_pos, color="lightgray", linestyle="--", linewidth=0.5, zorder=0)
                    if row_idx == 0 and i in (1, 2, 3, 4):
                        ax.text(x_pos, 1.02, f"{int(frac * 100)}%", transform=ax.get_xaxis_transform(),
                                ha="center", va="bottom", fontsize=6, color="gray")
                tick_positions, tick_labels, seen = [], [], set()
                for frac in perc_fracs:
                    org_count = int(np.floor(frac * max_x))
                    if org_count >= 0 and org_count not in seen:
                        seen.add(org_count); tick_positions.append(org_count); tick_labels.append(str(org_count))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
                ax.tick_params(axis="x", labelsize=6, colors="gray")
                for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
                if row_idx == 0: ax.set_title(cls, fontsize=11, y=1.2)
            row_ax = axes[row_idx, 0]
            row_ax.set_ylabel(country_name, rotation=90, fontsize=9, labelpad=20, va="center", ha="center")
            # Removed set_label_coords so Matplotlib places it dynamically outside the tick labels

        # Force all left-column y-labels to align perfectly along the same vertical line
        fig.align_ylabels(axes[:, 0])

        plt.tight_layout(rect=[0.15, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=1.0, wspace=1)
        
        if output_path:
            base, ext = os.path.splitext(output_path)
            part_path = f"{base}_{suffix}{ext}"
            fig.savefig(part_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {part_path}")
        plt.show()


# ===========================================================================
# Finetuner Diagnostics
# ===========================================================================

# Consistent colour per label class
_CLASS_COLORS = {
    "hospital": "#e6194b",
    "university_hospital": "#f58231",
    "local_government": "#4363d8",
    "primary_school": "#3cb44b",
    "secondary_school": "#911eb4",
    "negative": "#aaaaaa",
}
_LABEL_COLS = [
    "hospital",
    "university_hospital",
    "local_government",
    "primary_school",
    "secondary_school",
]


def _sample_probe_set(
    df: pd.DataFrame,
    n_per_class: int = 8,
    n_negative: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a small fixed probe set: n_per_class samples per labelled class,
    plus n_negative "all-negative" rows.

    Strategy (per class):
    1. Prefer rows that are *exclusively* in this class (no other label active).
    2. If fewer than n_per_class exclusive rows exist (e.g. university_hospital
       almost always co-occurs with hospital), fall back to any row where
       cls == 1, regardless of other labels.

    Returns a DataFrame with a 'probe_class' column.
    """
    rng = np.random.RandomState(seed)
    parts = []

    for cls in _LABEL_COLS:
        # --- Try exclusive mask first ---
        excl_mask = (df[cls] == 1)
        for other in _LABEL_COLS:
            if other != cls:
                excl_mask = excl_mask & (df[other] == 0)
        excl_pool = df[excl_mask]

        if len(excl_pool) >= n_per_class:
            pool = excl_pool
        else:
            # Fallback: any row where this class is positive
            pool = df[df[cls] == 1]
            if len(excl_pool) < len(pool):
                print(
                    f"[probe] '{cls}': only {len(excl_pool)} exclusive rows — "
                    f"falling back to any-positive pool ({len(pool)} rows)."
                )

        n = min(n_per_class, len(pool))
        if n == 0:
            print(f"[probe] '{cls}': no samples found, skipping.")
            continue
        sample = pool.sample(n=n, random_state=int(rng.randint(0, 9999))).copy()
        sample["probe_class"] = cls
        parts.append(sample)

    # All-negative rows
    neg_mask = (df[_LABEL_COLS] == 0).all(axis=1)
    neg_pool = df[neg_mask]
    n_neg = min(n_negative, len(neg_pool))
    if n_neg > 0:
        neg_sample = neg_pool.sample(n=n_neg, random_state=int(rng.randint(0, 9999))).copy()
        neg_sample["probe_class"] = "negative"
        parts.append(neg_sample)

    probe = pd.concat(parts, ignore_index=True)
    probe["names"] = probe["names"].astype(str)
    return probe


def _encode_probe(
    model,
    tokenizer,
    probe_df: pd.DataFrame,
    max_length: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode the probe DataFrame using mean-pool + L2-norm, same as finetuner.
    Returns (N, D) numpy array.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    texts = ["query: " + name for name in probe_df["names"].tolist()]
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normed = F.normalize(pooled, p=2, dim=-1)
            all_embs.append(normed.cpu().numpy())
    return np.vstack(all_embs)


def _project_2d(embeddings_per_epoch: list[np.ndarray]) -> list[np.ndarray]:
    """
    Fit a single PCA on the concatenation of all epochs' embeddings so that
    all epochs share the same 2-D coordinate system (traces are comparable).
    """
    from sklearn.decomposition import PCA

    all_emb = np.vstack(embeddings_per_epoch)          # (E*N, D)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_emb)
    return [pca.transform(e) for e in embeddings_per_epoch]  # list of (N, 2)


def _render_diagnostics(
    ax_loss,
    ax_proj,
    losses: list[dict],
    projections: list[np.ndarray],
    probe_classes: list[str],
    title_suffix: str = "",
) -> None:
    """
    Draw / refresh the two-panel figure in-place.
    """
    # ── Left panel: loss curves ───────────────────────────────────────────
    ax_loss.cla()
    epochs = [row["epoch"] for row in losses]
    train_losses = [row["train_loss"] for row in losses]
    val_losses = [row["val_loss"] for row in losses]

    ax_loss.plot(epochs, train_losses, "o-", color="steelblue", label="Train loss")
    ax_loss.plot(epochs, val_losses, "s--", color="coral", label="Val loss")
    best_epoch = losses[np.argmin(val_losses)]["epoch"]
    ax_loss.axvline(best_epoch, color="coral", linewidth=0.8, linestyle=":")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("SupCon loss")
    ax_loss.set_title(f"Loss curves{title_suffix}")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    # ── Right panel: 2-D embedding traces ────────────────────────────────
    ax_proj.cla()

    # Limit to first 8 epochs if more exist
    projections = projections[:8]
    if losses:
        losses = losses[:8]
    n_epochs = len(projections)

    unique_classes = list(dict.fromkeys(probe_classes))  # preserve order

    for cls in unique_classes:
        idxs = [i for i, c in enumerate(probe_classes) if c == cls]
        color = _CLASS_COLORS.get(cls, "black")

        # Draw markers and linking lines progressively fading up to the latest epoch
        for pt_idx in idxs:
            for e in range(n_epochs - 1):
                # alpha increases as we approach the latest epoch
                alpha_val = 0.1 + 0.8 * ((e + 1) / max(n_epochs - 1, 1))

                # Super thin line linking to the next epoch
                p1 = projections[e][pt_idx]
                p2 = projections[e + 1][pt_idx]
                ax_proj.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color=color,
                    alpha=alpha_val,
                    linewidth=0.5,
                    zorder=1
                )

                # Point marker at epoch e
                ax_proj.scatter(
                    p1[0],
                    p1[1],
                    color=color,
                    marker="x",
                    s=15,
                    alpha=alpha_val,
                    zorder=2,
                )

            # Only plot the point for the LATEST epoch
            if n_epochs > 0:
                ax_proj.scatter(
                    projections[-1][pt_idx, 0],
                    projections[-1][pt_idx, 1],
                    color=color,
                    marker="o",
                    s=15,
                    zorder=10,
                    edgecolors="none",
                )

    # Epoch gradient legend stub
    legend_handles = [
        Line2D([0], [0], marker="o", color=_CLASS_COLORS.get(cls, "black"),
               label=cls, markersize=5, linewidth=0)
        for cls in unique_classes
    ]
    ax_proj.legend(
        handles=legend_handles,
        fontsize=6,
        loc="lower right",
        framealpha=0.7,
    )
    ax_proj.set_title(
        f"2-D PCA projection (epoch 1 faint → epoch {n_epochs} bold){title_suffix}"
    )
    ax_proj.set_xlabel("PC 1")
    ax_proj.set_ylabel("PC 2")
    ax_proj.grid(True, alpha=0.2)


def watch_training_diagnostics(
    checkpoint_dir: str,
    data_path: str,
    n_per_class: int = 8,
    n_negative: int = 8,
    max_length: int = 128,
    poll_interval: float = 20.0,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> None:
    """
    Parallel diagnostic monitor for a running finetuner.py training session.

    Polls ``checkpoint_dir`` for new ``epoch_NNN/`` subdirectories and a
    ``training_log.jsonl`` file.  For each new epoch checkpoint it:

    1. Loads the HuggingFace model + tokenizer.
    2. Encodes a fixed probe set (sampled once at startup).
    3. Projects all epochs' embeddings to 2-D PCA (shared axis).
    4. Renders / refreshes a two-panel figure:
          Left  – train / val SupCon loss curves
          Right – 2-D scatter with per-point traces across epochs

    Can also be run **post-hoc** when training has already completed.

    Parameters
    ----------
    checkpoint_dir : str
        ``--output_dir`` used by finetuner.py (the dir that contains
        ``epoch_001/``, ``epoch_002/``, … and ``training_log.jsonl``).
    data_path : str
        CSV used for training (same as ``--data_path`` in finetuner.py).
        Used to sample the probe set.
    n_per_class : int
        How many exclusively-labelled examples to sample per class.
    n_negative : int
        How many "all-negative" examples to include.
    max_length : int
        Tokenizer max-length (match the value used in training).
    poll_interval : float
        Seconds between directory polls (ignored once all epochs are done).
    seed : int
        Random seed for reproducible probe sampling.
    output_path : str | None
        If set, save the final figure here (PNG/PDF) instead of
        / in addition to showing it interactively.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # ── Load data and build probe set once ───────────────────────────────
    print(f"[diagnostics] Loading data from {data_path} ...")
    try:
        df = pd.read_csv(data_path)
        for col in _LABEL_COLS:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["names"])
    except Exception as e:
        print(f"[diagnostics] ERROR loading data: {e}")
        return

    probe_df = _sample_probe_set(df, n_per_class=n_per_class, n_negative=n_negative, seed=seed)
    probe_classes = probe_df["probe_class"].tolist()
    print(
        f"[diagnostics] Probe set: {len(probe_df)} samples across "
        f"{len(set(probe_classes))} categories."
    )

    # ── Matplotlib setup ──────────────────────────────────────────────────
    matplotlib.use("Agg")          # safe for non-interactive / parallel use
    fig, (ax_loss, ax_proj) = plt.subplots(1, 2, figsize=(14, 6))
    fig.tight_layout(pad=3.0)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[diagnostics] Device: {device}")

    processed_epochs: set[int] = set()
    embeddings_per_epoch: list[np.ndarray] = []
    losses: list[dict] = []
    prev_loss_len = -1
    warned_log_missing = False

    log_path = os.path.join(checkpoint_dir, "training_log.jsonl")

    print(
        f"[diagnostics] Watching {checkpoint_dir} "
        f"(poll every {poll_interval}s) ..."
    )

    while True:
        # ── Read loss log ────────────────────────────────────────────────
        current_losses: list[dict] = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            current_losses.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            
            if len(current_losses) == 0 and not warned_log_missing:
                print(f"[diagnostics] WARNING: Log file is empty: {log_path}")
                warned_log_missing = True
        else:
            if not warned_log_missing:
                print(f"[diagnostics] WARNING: Log file not found at: {log_path}")
                warned_log_missing = True

        # ── Detect new epoch checkpoints ─────────────────────────────────
        new_epochs_found = False
        if os.path.isdir(checkpoint_dir):
            for entry in sorted(os.listdir(checkpoint_dir)):
                if not entry.startswith("epoch_"):
                    continue
                try:
                    epoch_num = int(entry.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if epoch_num in processed_epochs:
                    continue
                if epoch_num > 8:
                    continue

                epoch_dir = os.path.join(checkpoint_dir, entry)
                # Check that the checkpoint is fully written
                if not os.path.exists(os.path.join(epoch_dir, "config.json")):
                    continue

                print(f"[diagnostics] Loading checkpoint epoch {epoch_num} ...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(epoch_dir)
                    model = AutoModel.from_pretrained(epoch_dir).to(device)
                    emb = _encode_probe(model, tokenizer, probe_df, max_length=max_length)
                    embeddings_per_epoch.append(emb)
                    processed_epochs.add(epoch_num)
                    new_epochs_found = True
                    # Free GPU memory
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[diagnostics] Could not load epoch {epoch_num}: {e}")

        # ── Refresh plot if anything new ─────────────────────────────────
        log_changed = len(current_losses) != prev_loss_len
        if (new_epochs_found or log_changed) and len(embeddings_per_epoch) >= 1:
            prev_loss_len = len(current_losses)
            losses = current_losses or [
                {"epoch": i + 1, "train_loss": 0.0, "val_loss": 0.0, "is_best": False}
                for i in range(len(embeddings_per_epoch))
            ]
            # Only keep loss rows for epochs we've processed
            n_proc = len(embeddings_per_epoch)
            losses_trimmed = losses[:n_proc]

            projections = _project_2d(embeddings_per_epoch)

            _render_diagnostics(
                ax_loss, ax_proj,
                losses=losses_trimmed,
                projections=projections,
                probe_classes=probe_classes,
            )
            fig.tight_layout(pad=3.0)

            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"[diagnostics] Figure saved to {output_path}")

        # ── Decide whether to keep polling ───────────────────────────────
        # Stop if: log exists AND we have processed all logged epochs
        if current_losses and processed_epochs:
            last_logged_epoch = max(r["epoch"] for r in current_losses)
            if last_logged_epoch in processed_epochs:
                # Check whether training terminated (early stopping or max epochs)
                # by seeing if the log hasn't grown for one more poll
                print(
                    f"[diagnostics] All {len(processed_epochs)} logged epochs processed. "
                    "Waiting one more cycle to confirm training is done..."
                )
                time.sleep(poll_interval)
                # Re-read log
                new_log: list[dict] = []
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    new_log.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                if len(new_log) == len(current_losses):
                    print("[diagnostics] Training appears complete. Exiting poll loop.")
                    break

        time.sleep(poll_interval)

    # ── Final figure ──────────────────────────────────────────────────────
    if embeddings_per_epoch:
        projections = _project_2d(embeddings_per_epoch)
        _render_diagnostics(
            ax_loss, ax_proj,
            losses=losses,
            projections=projections,
            probe_classes=probe_classes,
        )
        fig.tight_layout(pad=3.0)
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[diagnostics] Final figure saved to {output_path}")

    return fig