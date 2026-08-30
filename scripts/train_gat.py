"""Trains the sector-graph GAT (Experiment 4): PriceEncoder + 2-layer GAT
over same-sector edges. Reports Accuracy/F1/MCC/AUC on the exact-
reproduction test split, in the same schema as Experiments 1 and 3.

Reminder before citing results: this is a labeled proxy for MAN-SF's
GAT+price component (sector edges, not their Wikidata company-relation
graph) -- see
docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md.

GRAD_ACCUM_DAYS: root-cause fix for a training collapse diagnosed via
systematic debugging (controlled ablation across GAT/Transformer/no-graph
variants -- all collapsed identically regardless of architecture). One
gradient step per single trading day gives the optimizer near-zero
diversity per step, since every ticker active on a given day shares that
day's market-wide conditions -- unlike Experiment 3's randomly-shuffled
batch_size=64 sampling across unrelated tickers/dates. Accumulating
gradients over GRAD_ACCUM_DAYS days before each optimizer step (still
respecting per-day snapshot/graph boundaries) restored real, non-collapsing
learning in a minimal ablation before this fix was applied here.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots
from src.models.graph_model import StockGAT, build_sector_graph, edge_index_for_tickers
from src.utils.seed import set_seed

PARQUET = "dataset/stocknet_protocol_a_exact.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
OUT_DIR = "results/experiment4_gat"
PATIENCE = 10
MAX_EPOCHS = 100
LR = 1e-3
GRAD_ACCUM_DAYS = 16


def run_epoch(model, snapshots, graph, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        # Shuffle the day-list order each training epoch (not the tickers
        # within a day) so gradient steps aren't seen in the same
        # chronological order every epoch. Dev/test order is left as-is --
        # order doesn't affect evaluation.
        snapshots = list(snapshots)
        random.shuffle(snapshots)
    all_probs, all_targets = [], []
    total_loss = 0.0
    if is_train:
        optimizer.zero_grad()
    for i, snap in enumerate(snapshots):
        edge_index = edge_index_for_tickers(graph, snap.tickers)
        logits = model(snap.price_seq, snap.price_mask, edge_index)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            (loss / GRAD_ACCUM_DAYS).backward()
            if (i + 1) % GRAD_ACCUM_DAYS == 0 or i == len(snapshots) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * len(snap.tickers)
        all_probs.append(torch.softmax(logits, dim=-1)[:, 1].detach())
        all_targets.append(snap.target)
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "auc": roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.5,
    }
    return total_loss / len(targets), metrics


def main():
    set_seed(42)
    df = pd.read_parquet(PARQUET)
    graph = build_sector_graph(SECTOR_CSV)
    train_snaps = build_daily_snapshots(df, "train")
    dev_snaps = build_daily_snapshots(df, "dev")
    test_snaps = build_daily_snapshots(df, "test")
    normalize_price_snapshots(train_snaps, dev_snaps, test_snaps)
    n_samples = sum(len(s.tickers) for s in test_snaps)

    model = StockGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_mcc = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        train_loss, train_metrics = run_epoch(model, train_snaps, graph, optimizer)
        _, val_metrics = run_epoch(model, dev_snaps, graph)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"train_mcc={train_metrics['mcc']:.4f} val_mcc={val_metrics['mcc']:.4f}")
        if val_metrics["mcc"] > best_val_mcc:
            best_val_mcc = val_metrics["mcc"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    _, test_metrics = run_epoch(model, test_snaps, graph)
    print("Test metrics:", test_metrics)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(
            {"test": {
                "label": "Sector-graph GAT (price only)",
                "backend": None,
                **test_metrics,
                "n_samples": n_samples,
                "best_epoch": best_epoch,
                "best_val_mcc": best_val_mcc,
            }},
            f, indent=2,
        )
    print(f"Saved {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
