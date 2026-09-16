"""Experiment 10 -- the multi-modal graph, built properly and measured honestly.

What is new versus Experiments 4-7:

  * Node features carry ALL modalities at once: price window, EDGAR fundamentals,
    tweet counts, compressed tweet text, and explicit volatility estimates.
    No previous experiment did this -- 4/5/6 were price-only, 7 had no fundamentals.
  * The volatility features (EWMA, realised 5/10/20d) put the project's strongest
    measured signal ON the nodes, so the graph is asked to add to the best thing
    available rather than to price alone.
  * `no_graph` runs the identical network with message passing off, in the SAME
    harness. Experiment 6 lacked this and consequently could not tell that its
    best cell was calling market direction rather than ranking stocks.
  * Within-day MCC is reported for every cell and is the PRIMARY metric. Pooled
    MCC on this benchmark is roughly half market direction (a day-direction
    oracle scores +0.6248 with no stock-specific skill at all).
  * Both targets: direction, and volatility (where a real effect already exists
    and where cross-firm volatility spillover is a documented phenomenon).

Usage:
  --screen         one seed, every conv, both targets, graph vs no-graph
  --full CONV      five seeds on one conv plus its no-graph control
"""

import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.edge_cache import load_edge_cache
from src.data.graph_edges import combine_edges
from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots
from src.data.node_features import assemble, standardise
from src.models.multimodal_graph import CONV_TYPES, MultiModalGraphNet
from src.utils.metrics import pooled_metrics, within_day_metrics
from src.utils.seed import set_seed

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
TEXT_CACHE = "data/processed/experiment8_text_matrix.pt"
EDGE_CACHE = "results/experiment6_dense_graph/edge_cache.pkl"
OUT_DIR = "results/experiment10_multimodal"
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")

RELATIONS = ["sector", "correlation", "comention"]
TEXT_PCA_DIMS = 8
PATIENCE, MAX_EPOCHS, LR, GRAD_ACCUM_DAYS = 15, 60, 1e-3, 16
MIN_EPOCHS = 10
WINSORIZE_LIMIT = 10.0


def load_everything(target: str):
    df = pd.read_parquet(EXACT)
    df["day"] = pd.to_datetime(df["Target_Date"]).dt.strftime("%Y-%m-%d")

    if target == "volatility":
        magnitude = df["Main_MV_Percent"].abs()
        threshold = magnitude[df["Split"] == "train"].median()
        df["Target"] = (magnitude > threshold).astype(int)

    is_train = (df["Split"] == "train").to_numpy()

    text_pca = None
    if os.path.exists(TEXT_CACHE):
        matrix = torch.load(TEXT_CACHE, weights_only=False)["matrix"].numpy()
        if len(matrix) == len(df):
            centre = matrix[is_train].mean(axis=0)
            _, _, vt = np.linalg.svd(matrix[is_train] - centre, full_matrices=False)
            text_pca = (matrix - centre) @ vt[:TEXT_PCA_DIMS].T   # fit on train only
    static, names = assemble(df, FINAL, text_pca)
    static = standardise(static, is_train)
    print(f"  static node features: {static.shape[1]} "
          f"({'with' if text_pca is not None else 'WITHOUT'} text)")

    lookup = {(t, d): static[i] for i, (t, d) in
              enumerate(zip(df["Ticker"], df["day"]))}

    splits = tuple(build_daily_snapshots(df, s) for s in ("train", "dev", "test"))
    normalize_price_snapshots(*splits, winsorize_limit=WINSORIZE_LIMIT)

    cache = load_edge_cache(EDGE_CACHE)
    return splits, lookup, cache, static.shape[1], names


def static_for(snap, lookup, dim):
    return torch.tensor(np.stack(
        [lookup.get((t, snap.date), np.zeros(dim, dtype=np.float32))
         for t in snap.tickers]), dtype=torch.float32)


def run_epoch(model, snapshots, lookup, cache, dim, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    order = list(snapshots)
    if is_train:
        random.shuffle(order)
        optimizer.zero_grad()

    probs, targets, days, total = [], [], [], 0.0
    for i, snap in enumerate(order):
        day_edges = cache.get(snap.date, {})
        edge_index, edge_type = combine_edges(
            {r: day_edges.get(r, set()) for r in RELATIONS}, snap.tickers)
        logits = model(snap.price_seq, snap.price_mask,
                       static_for(snap, lookup, dim), edge_index, edge_type)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            (loss / GRAD_ACCUM_DAYS).backward()
            if (i + 1) % GRAD_ACCUM_DAYS == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        total += loss.item() * len(snap.tickers)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].detach())
        targets.append(snap.target)
        days.extend([snap.date] * len(snap.tickers))

    p = torch.cat(probs).numpy()
    y = torch.cat(targets).numpy()
    metrics = pooled_metrics(y, p)
    metrics.update(within_day_metrics(y, p, days))
    metrics["n_samples"] = int(len(y))
    return total / len(y), metrics


def train_one(conv: str, use_graph: bool, seed: int, splits, lookup, cache, dim,
              contrast: bool = False, cross_sectional: bool = False):
    set_seed(seed)
    train_s, dev_s, test_s = splits
    model = MultiModalGraphNet(static_dim=dim, conv_type=conv,
                               num_relations=len(RELATIONS), use_graph=use_graph,
                               contrast=contrast, cross_sectional=cross_sectional)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best, best_state, stale = -2.0, None, 0
    for epoch in range(MAX_EPOCHS):
        run_epoch(model, train_s, lookup, cache, dim, optimizer)
        _, val = run_epoch(model, dev_s, lookup, cache, dim)
        # Selection on WITHIN-DAY dev MCC, not pooled: the goal is ranking stocks
        # within a day, so selecting on pooled would optimise for calling the
        # market -- precisely the failure mode Experiment 6 fell into.
        score = val["within_day_mcc"]
        if score > best:
            best, stale = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    _, test = run_epoch(model, test_s, lookup, cache, dim)
    test["best_val_within_day_mcc"] = best
    return test


def record(row: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)


def done(target, conv, use_graph, seed, variant) -> bool:
    if not os.path.exists(RESULTS_CSV):
        return False
    d = pd.read_csv(RESULTS_CSV)
    if "variant" not in d.columns:
        d["variant"] = "plain"
    return ((d.target == target) & (d.conv == conv) & (d.variant == variant) &
            (d.use_graph == use_graph) & (d.seed == seed)).any()


# (name, contrast, cross_sectional). "plain" is the screen already run: standard
# smoothing message passing. The other three isolate the two fixes and their
# combination, so the write-up can say which one (if either) did the work.
VARIANTS = {
    "plain":        (False, False),
    "contrast":     (True, False),
    "xsect":        (False, True),
    "contrast+xs":  (True, True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--full", default=None, choices=list(CONV_TYPES))
    ap.add_argument("--targets", nargs="*", default=["direction", "volatility"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=["plain"],
                    help=f"any of {list(VARIANTS)}")
    args = ap.parse_args()

    if args.screen:
        convs, seeds = list(CONV_TYPES), [0]
    elif args.full:
        convs, seeds = [args.full], (args.seeds or [0, 1, 2, 3, 4])
    else:
        sys.exit("pass --screen or --full CONV")

    for target in args.targets:
        print(f"\n=== target: {target} ===", flush=True)
        splits, lookup, cache, dim, names = load_everything(target)
        for variant, conv, use_graph, seed in itertools.product(
                args.variants, convs, [True, False], seeds):
            if done(target, conv, use_graph, seed, variant):
                print(f"  skip {variant} {conv} graph={use_graph} seed={seed}")
                continue
            contrast, cross_sectional = VARIANTS[variant]
            # With the graph off, contrast is a no-op (it lives inside the conv
            # loop), so only cross-sectional changes anything -- skip the
            # duplicate cell rather than recording the same run twice.
            if not use_graph and contrast and not cross_sectional:
                continue
            tag = f"{variant:<12} {conv:<12} graph={str(use_graph):<5} seed={seed}"
            print(f"  train {tag} ...", flush=True)
            test = train_one(conv, use_graph, seed, splits, lookup, cache, dim,
                             contrast=contrast, cross_sectional=cross_sectional)
            record({"target": target, "variant": variant, "conv": conv,
                    "use_graph": use_graph, "seed": seed, **test})
            print(f"        {tag}  within-day={test['within_day_mcc']:+.4f}  "
                  f"pooled={test['mcc']:+.4f}  acc={test['accuracy']:.4f}  "
                  f"n={test['n_samples']}", flush=True)

    print(f"\nDone. {RESULTS_CSV}")


if __name__ == "__main__":
    main()
