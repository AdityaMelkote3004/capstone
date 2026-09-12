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
from src.utils.metrics import within_day_metrics
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

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
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


def edge_combos() -> list:
    """All 7 non-empty subsets of the three relations, shortest first."""
    out = []
    for r in (1, 2, 3):
        for combo in itertools.combinations(RELATIONS, r):
            out.append(list(combo))
    return out


def run_epoch(model: nn.Module, snapshots: list, cache: dict, combo: list,
              optimizer: torch.optim.Optimizer = None) -> tuple:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        snapshots = list(snapshots)
        random.shuffle(snapshots)
        optimizer.zero_grad()

    probs_all, targets_all, days_all, total_loss = [], [], [], 0.0
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
        days_all.extend([snap.date] * len(snap.tickers))

    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "auc": roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.5,
    }
    # Pooled MCC on this benchmark is ~half market direction (48.3% of test label
    # variance is between-day; a day-direction oracle scores +0.6248). Experiment 6
    # originally reported pooled only, so it could not distinguish "picks stocks"
    # from "calls the market". Within-day closes that gap.
    metrics.update(within_day_metrics(targets, probs, days_all))
    return total_loss / len(targets), metrics


def train_one(model_name: str, combo: list, seed: int, snaps: tuple, cache: dict) -> dict:
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


def already_done(model_name: str, combo: list, seed: int) -> bool:
    if not os.path.exists(RESULTS_CSV):
        return False
    try:
        df = pd.read_csv(RESULTS_CSV)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # A crash mid-write (Colab session death, OOM kill) can leave the
        # results file zero bytes or with a truncated last line. Treat that
        # as "nothing recorded yet" and re-run the cell rather than dying on
        # resume -- which is the exact scenario this skip logic exists for.
        return False
    df = df.dropna(subset=["model", "edges", "seed"])
    key = (df["model"] == model_name) & (df["edges"] == "+".join(combo)) & (df["seed"] == seed)
    return bool(key.any())


def record(model_name: str, combo: list, seed: int, test: dict) -> None:
    label = f"{model_name} [{'+'.join(combo)}] seed={seed}"
    run_dir = os.path.join(OUT_DIR, f"{model_name}__{'_'.join(combo)}__seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump({"test": {"label": label, "backend": None, **test}}, fh, indent=2)

    row = {"model": model_name, "edges": "+".join(combo), "seed": seed, **test}
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_num", type=int, default=1, choices=[1, 2, 3])
    # Experiment 6 reported its headline (+0.0677, RGCN sector+correlation) on
    # THREE seeds with a 0.0509 spread. Experiment 1's headline collapsed under a
    # 5-seed rerun, so these let one cell be topped up to 5 rather than re-running
    # the whole 84-cell grid.
    ap.add_argument("--only-model", default=None, help="restrict to one model")
    ap.add_argument("--only-combo", default=None,
                    help="restrict to one edge combo, e.g. sector+correlation")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override the pass's seed list")
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
    if args.seeds is not None:
        seeds = args.seeds
    if args.only_model:
        model_names = [args.only_model]
    combos = edge_combos()
    if args.only_combo:
        wanted = args.only_combo.split("+")
        combos = [c for c in combos if c == wanted]
        if not combos:
            sys.exit(f"no edge combo matches {args.only_combo!r}; "
                     f"valid relations are {RELATIONS}")
    todo = [(m, c, s) for m in model_names for c in combos for s in seeds]
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
