"""Runs Experiment 6's escalating grid: every non-empty combination of
{sector, correlation, comention} edges, across four graph models and
multiple seeds.

Results are appended after EVERY run and completed cells are skipped on
restart, so a crashed Colab session resumes where it stopped. Pass --pass 1,
2, or 3 to control scope; each pass is a superset of the previous one and
re-runs nothing.
"""

import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.data.edge_cache import build_edge_cache, load_edge_cache
from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots
from src.data.graph_edges import combine_edges
from src.models.graph_model import StockGAT
from src.models.graph_models import StockGraphSAGE, StockRGCN
from src.models.graph_transformer_multi import MultiRelationGraphTransformer
from src.utils.seed import set_seed

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
OUT_DIR = "results/experiment6_dense_graph"
EDGE_CACHE = os.path.join(OUT_DIR, "edge_cache.pkl")
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")

PATIENCE, MAX_EPOCHS, LR, GRAD_ACCUM_DAYS = 10, 100, 1e-3, 16
RELATIONS = ["sector", "correlation", "comention"]


class GATAdapter(nn.Module):
    """Experiment 4's StockGAT.forward takes (price_seq, price_mask,
    edge_index) with no edge_type. This adapter gives it the same 4-argument
    signature as every other model so the grid runner can call them
    identically, without editing StockGAT (whose published Experiment 4
    results must stay reproducible)."""

    def __init__(self):
        super().__init__()
        self.inner = StockGAT()

    def forward(self, price_seq, price_mask, edge_index, edge_type=None):
        return self.inner(price_seq, price_mask, edge_index)


MODELS = {
    "GAT": lambda: GATAdapter(),
    "GraphTransformer": lambda: MultiRelationGraphTransformer(),
    "GraphSAGE": lambda: StockGraphSAGE(),
    "RGCN": lambda: StockRGCN(),
}
PASS_SPEC = {
    1: (["GAT", "GraphTransformer"], [0]),
    2: (["GAT", "GraphTransformer"], [0, 1, 2]),
    3: (list(MODELS), [0, 1, 2]),
}


def edge_combos():
    """All 7 non-empty subsets of the three relations, shortest first."""
    out = []
    for r in (1, 2, 3):
        for combo in itertools.combinations(RELATIONS, r):
            out.append(list(combo))
    return out


def run_epoch(model, snapshots, cache, combo, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        snapshots = list(snapshots)
        random.shuffle(snapshots)
        optimizer.zero_grad()

    probs_all, targets_all, total_loss = [], [], 0.0
    for i, snap in enumerate(snapshots):
        day = cache.get(snap.date, {})
        edge_sets = {r: day.get(r, set()) for r in combo}
        edge_index, edge_type = combine_edges(edge_sets, snap.tickers)
        logits = model(snap.price_seq, snap.price_mask, edge_index, edge_type)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            (loss / GRAD_ACCUM_DAYS).backward()
            if (i + 1) % GRAD_ACCUM_DAYS == 0 or i == len(snapshots) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * len(snap.tickers)
        probs_all.append(torch.softmax(logits, dim=-1)[:, 1].detach())
        targets_all.append(snap.target)

    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy()
    preds = (probs >= 0.5).astype(int)
    return total_loss / len(targets), {
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "auc": roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.5,
    }


def train_one(model_name, combo, seed, snaps, cache):
    set_seed(seed)
    train_snaps, dev_snaps, test_snaps = snaps
    model = MODELS[model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_mcc, best_epoch, best_state, stale = -1.0, -1, None, 0
    for epoch in range(MAX_EPOCHS):
        _, _ = run_epoch(model, train_snaps, cache, combo, optimizer)
        _, val = run_epoch(model, dev_snaps, cache, combo)
        if val["mcc"] > best_mcc:
            best_mcc, best_epoch, stale = val["mcc"], epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    _, test = run_epoch(model, test_snaps, cache, combo)
    test["n_samples"] = sum(len(s.tickers) for s in test_snaps)
    test["best_epoch"] = best_epoch
    test["best_val_mcc"] = best_mcc
    return test


def already_done(model_name, combo, seed):
    if not os.path.exists(RESULTS_CSV):
        return False
    df = pd.read_csv(RESULTS_CSV)
    key = (df["model"] == model_name) & (df["edges"] == "+".join(combo)) & (df["seed"] == seed)
    return bool(key.any())


def record(model_name, combo, seed, test):
    label = f"{model_name} [{'+'.join(combo)}] seed={seed}"
    run_dir = os.path.join(OUT_DIR, f"{model_name}__{'_'.join(combo)}__seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump({"test": {"label": label, "backend": None, **test}}, fh, indent=2)

    row = {"model": model_name, "edges": "+".join(combo), "seed": seed, **test}
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_num", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(EDGE_CACHE):
        cache = load_edge_cache(EDGE_CACHE)
        print(f"Loaded edge cache ({len(cache)} days)")
    else:
        print("Building edge cache (one-time, ~1-2 min)...")
        cache = build_edge_cache(EXACT, FINAL, SECTOR_CSV, EDGE_CACHE)

    df = pd.read_parquet(EXACT)
    train_snaps = build_daily_snapshots(df, "train")
    dev_snaps = build_daily_snapshots(df, "dev")
    test_snaps = build_daily_snapshots(df, "test")
    normalize_price_snapshots(train_snaps, dev_snaps, test_snaps)
    snaps = (train_snaps, dev_snaps, test_snaps)

    model_names, seeds = PASS_SPEC[args.pass_num]
    todo = [(m, c, s) for m in model_names for c in edge_combos() for s in seeds]
    print(f"Pass {args.pass_num}: {len(todo)} cells in grid")

    for model_name, combo, seed in todo:
        if already_done(model_name, combo, seed):
            print(f"  skip  {model_name} [{'+'.join(combo)}] seed={seed}")
            continue
        print(f"  train {model_name} [{'+'.join(combo)}] seed={seed} ...", flush=True)
        test = train_one(model_name, combo, seed, snaps, cache)
        record(model_name, combo, seed, test)
        print(f"        MCC={test['mcc']:+.4f} F1={test['f1']:.4f} AUC={test['auc']:.4f}")

    print(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
