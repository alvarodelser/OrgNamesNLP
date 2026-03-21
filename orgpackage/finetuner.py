"""
finetuner.py
============
Fine-tunes intfloat/multilingual-e5-base on the Wikidata organisation-name
dataset using multi-label Supervised Contrastive Loss (SupCon).

Usage:
    python -m orgpackage.finetuner [options]
    python orgpackage/finetuner.py [options]

References:
    Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020.
    https://arxiv.org/abs/2004.11362
"""

from __future__ import annotations

import argparse
import json
import ast
import copy
import os
import random
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from orgpackage.aux import load_dataset

# ---------------------------------------------------------------------------
# Label columns (multi-label, binary)
# ---------------------------------------------------------------------------
LABEL_COLS: List[str] = [
    "hospital",
    "university_hospital",
    "local_government",
    "primary_school",
    "secondary_school",
]


# ===========================================================================
# 1. Dataset
# ===========================================================================

class OrgNameDataset(Dataset):
    """Wraps a DataFrame of organisation names with multi-hot labels."""

    def __init__(self, df: pd.DataFrame, label_cols: List[str] = LABEL_COLS):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        # mE5 requires the "query: " prefix for names to be embedded
        text = "query: " + str(row["names"])
        labels = torch.tensor(
            [float(row[c]) for c in self.label_cols], dtype=torch.float32
        )
        country = str(row.get("country", "unknown"))
        return {"text": text, "labels": labels, "country": country}


# ===========================================================================
# 2. Balanced Batch Sampler (best-effort per country)
# ===========================================================================

class BalancedBatchSampler:
    """
    Yields batch indices by interleaving countries in round-robin order
    so that each batch is as country-diverse as possible.

    No hard minimum per country is enforced – very small countries simply
    contribute fewer samples.
    """

    def __init__(self, dataset: OrgNameDataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        # Build country → [indices] mapping
        country_to_indices: dict[str, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            country = dataset.df.iloc[i]["country"]
            country_to_indices[str(country)].append(i)
        self.country_to_indices = dict(country_to_indices)
        self.countries = sorted(self.country_to_indices.keys())

    def __iter__(self):
        # Shuffle each country's pool at the start of every epoch
        pools = {
            c: self.rng.sample(idxs, len(idxs))
            for c, idxs in self.country_to_indices.items()
        }
        pointers = {c: 0 for c in self.countries}

        total = len(self.dataset)
        yielded = 0
        country_cycle = list(self.countries)

        while yielded < total:
            batch: list[int] = []
            c_idx = 0
            # Round-robin across countries until batch is full
            full_rounds = 0
            while len(batch) < self.batch_size and full_rounds < 2:
                added_in_pass = 0
                for c in country_cycle:
                    if len(batch) >= self.batch_size:
                        break
                    p = pointers[c]
                    if p < len(pools[c]):
                        batch.append(pools[c][p])
                        pointers[c] += 1
                        added_in_pass += 1
                if added_in_pass == 0:
                    full_rounds += 1  # All pools exhausted

            if batch:
                self.rng.shuffle(batch)
                yield batch
                yielded += len(batch)
            else:
                break

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ===========================================================================
# 3. Encoder helpers
# ===========================================================================

def mean_pool_normalize(
    model_output, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool last hidden state over non-padding tokens, then L2-normalise."""
    last_hidden = model_output.last_hidden_state          # (B, L, D)
    mask = attention_mask.unsqueeze(-1).float()           # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)              # (B, D)
    counts = mask.sum(dim=1).clamp(min=1e-9)              # (B, 1)
    pooled = summed / counts                              # (B, D)
    return F.normalize(pooled, p=2, dim=-1)               # (B, D)


def encode_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize → encode → mean-pool → L2-normalise."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.set_grad_enabled(model.training):
        output = model(**encoded)
    return mean_pool_normalize(output, encoded["attention_mask"])


# ===========================================================================
# 4. Multi-label Supervised Contrastive Loss
# ===========================================================================

class MultiLabelSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for multi-label classification.

    Positive pair definition
    ------------------------
    (i, j) is positive iff they share at least one active label
    (label_i[c] == 1 AND label_j[c] == 1 for some c).

    "All-negative" rows (none of the K labels active) never form positive
    pairs with any other sample, but they still contribute to the denominator
    of every other anchor's softmax, acting as hard negatives that push the
    labelled clusters tighter.  Their own anchor term emits zero loss
    (no positives to sum over).

    Reference: Khosla et al. NeurIPS 2020, eq. (2).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features : (N, D)  L2-normalised embeddings.
            labels   : (N, K)  multi-hot binary label matrix.
        Returns:
            Scalar loss.
        """
        N = features.size(0)
        device = features.device

        if N < 2:
            return features.new_tensor(0.0)

        # ── Similarity matrix ─────────────────────────────────────────────
        # sim[i, j] = cosine similarity (features already L2-normalised)
        sim = torch.mm(features, features.T) / self.temperature  # (N, N)

        # For numerical stability subtract row-max before exp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # ── Build positive mask ───────────────────────────────────────────
        # pos_mask[i, j] = 1 iff i and j share at least one active class
        # = 1 iff (labels[i] * labels[j]).sum() > 0
        # We keep the batch-matrix product in float32
        shared = torch.mm(labels.float(), labels.float().T)  # (N, N)
        pos_mask = (shared > 0).float()                       # (N, N)

        # Exclude self-similarities from both pos_mask and denominator
        self_mask = torch.eye(N, device=device)
        pos_mask = pos_mask * (1.0 - self_mask)

        # ── Denominator: all pairs except self ────────────────────────────
        denom_mask = 1.0 - self_mask                          # (N, N)
        exp_sim = torch.exp(sim) * denom_mask                 # zero out diagonal

        # ── Numerator: positive pairs ─────────────────────────────────────
        # log P(positive | anchor) = sim_pos - log(sum_denom)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)  # (N,1)
        log_prob = sim - log_denom                              # (N, N)

        # ── Per-anchor loss ───────────────────────────────────────────────
        # anchors without any positive in this batch → skip (zero contribution)
        num_pos = pos_mask.sum(dim=1)                         # (N,)
        has_pos = (num_pos > 0).float()                       # (N,)

        # Mean log-prob over positive pairs per anchor
        anchor_loss = -(pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-9)

        # Only count anchors that have at least one positive
        loss = (has_pos * anchor_loss).sum() / (has_pos.sum() + 1e-9)
        return loss


# ===========================================================================
# 5. Data loading
# ===========================================================================

def prepare_finetuning_data(df: pd.DataFrame, label_cols: List[str] = LABEL_COLS) -> pd.DataFrame:
    """
    Ensure required label columns exist, enforce numeric types,
    and drop non-usable rows (missing names).
    """
    df = df.copy()
    
    # Ensure label columns exist (default 0 if missing)
    for col in label_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Drop rows with missing names
    df = df.dropna(subset=["names"])
    df["names"] = df["names"].astype(str)

    print(f"[finetuner] Data prepared. Total rows: {len(df):,}")
    pos_counts = {c: int(df[c].sum()) for c in label_cols}
    print(f"[finetuner] Label counts: {pos_counts}")
    return df


def train_val_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
    label_cols: List[str] = LABEL_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Global 80/20 stratified split.  Stratification key is the bitmask of
    active label columns so class proportions are preserved.
    """
    df = df.copy()
    df["_strat_key"] = df[label_cols].apply(
        lambda row: "".join(str(int(v)) for v in row), axis=1
    )
    train_parts, val_parts = [], []
    rng = np.random.RandomState(seed)
    for _, group in df.groupby("_strat_key", sort=False):
        idx = group.index.tolist()
        rng.shuffle(idx)
        split = max(1, int(len(idx) * train_ratio))
        train_parts.append(df.loc[idx[:split]])
        val_parts.append(df.loc[idx[split:]])
    train_df = pd.concat(train_parts).drop(columns=["_strat_key"])
    val_df = pd.concat(val_parts).drop(columns=["_strat_key"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ===========================================================================
# 6. Validation loss helper
# ===========================================================================

def compute_val_loss(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    val_dataset: OrgNameDataset,
    criterion: MultiLabelSupConLoss,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    model.eval()
    sampler = BalancedBatchSampler(val_dataset, batch_size=batch_size)
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch_indices in sampler:
            items = [val_dataset[i] for i in batch_indices]
            texts = [it["text"] for it in items]
            labels = torch.stack([it["labels"] for it in items]).to(device)
            features = encode_batch(model, tokenizer, texts, max_length, device)
            loss = criterion(features, labels)
            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1
    return total_loss / max(n_batches, 1)


# ===========================================================================
# 7. Training loop
# ===========================================================================

def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str = "results/finetuned_me5",
    model_name: str = "intfloat/multilingual-e5-base",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 2e-5,
    temperature: float = 0.07,
    max_length: int = 128,
    patience: int = 3,
    seed: int = 42,
    save_epochs: bool = True,
) -> None:
    """
    Orchestrate finetuning of mE5-base using Multi-label SupCon Loss.
    """
    # ── Reproducibility ───────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetuner] Device: {device}")
    
    print(
        f"[finetuner] Train: {len(train_df):,} rows | Val: {len(val_df):,} rows"
    )

    # Where metrics get written (one JSON line per epoch)
    log_path = os.path.join(output_dir, "training_log.jsonl")
    # Wipe any old log from a previous run
    if os.path.exists(log_path):
        os.remove(log_path)

    train_dataset = OrgNameDataset(train_df)
    val_dataset = OrgNameDataset(val_df)

    # ── Model & tokenizer ─────────────────────────────────────────────────
    print(f"[finetuner] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # ── Loss, optimiser ───────────────────────────────────────────────────
    criterion = MultiLabelSupConLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ── Training ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    patience_counter = 0

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        sampler = BalancedBatchSampler(
            train_dataset, batch_size=batch_size, seed=seed + epoch
        )
        total_train_loss = 0.0
        n_train_batches = 0

        for batch_idx, batch_indices in enumerate(sampler):
            items = [train_dataset[i] for i in batch_indices]
            texts = [it["text"] for it in items]
            labels = torch.stack([it["labels"] for it in items]).to(device)

            optimizer.zero_grad()
            features = encode_batch(
                model, tokenizer, texts, max_length, device
            )
            loss = criterion(features, labels)

            if torch.isnan(loss):
                print(f"  [warn] NaN loss at batch {batch_idx}, skipping.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        avg_val_loss = compute_val_loss(
            model, tokenizer, val_dataset, criterion,
            batch_size, max_length, device,
        )

        is_best = avg_val_loss < best_val_loss - 1e-6
        print(
            f"[epoch {epoch:03d}/{epochs}]  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}"
            + ("  ← best" if is_best else "")
        )

        # ── Per-epoch checkpoint ──────────────────────────────────────────
        if save_epochs:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

        # ── Early stopping / best model tracking ──────────────────────────
        if is_best:
            best_val_loss = avg_val_loss
            # clone state dict directly on CPU
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[finetuner] Early stopping triggered after {epoch} epochs "
                    f"(patience={patience})."
                )
                break

    # ── Save best checkpoint ──────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[finetuner] Model saved to: {output_dir}")
    print(f"[finetuner] Best validation loss: {best_val_loss:.4f}")
