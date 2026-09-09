"""Runs Experiment 7's 8-row ablation ladder at 5 seeds (40 runs).

Results are appended after EVERY run and completed cells are skipped on
restart, so an interrupted session resumes where it stopped.

Every row reports pooled AND within-day metrics: 48.3% of test label variance
is between-day, so pooled MCC alone cannot distinguish "discriminates stocks"
from "predicts market direction".
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn

from src.data.graph_edges import combine_edges
from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots
from src.data.macro_encoder import build_macro_vectors, load_macro_calendar
from src.data.micro_encoder import build_micro_embeddings, load_ticker_tweets
from src.data.tweet_graph import build_tweet_edges
from src.models.finbert_cache import EmbeddingCache
from src.models.graph_model import build_sector_graph, edge_index_for_tickers
from src.models.macro_conditioned import MacroConditionedGraphNet
from src.utils.metrics import pooled_metrics, within_day_metrics
from src.utils.seed import set_seed

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
CALENDAR = "dataset/macro_calendar.csv"
CACHE_PATH = "data/processed/experiment7_embeddings.pt"
OUT_DIR = "results/experiment7_macro_conditioned"
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")

PATIENCE, MAX_EPOCHS, LR, GRAD_ACCUM_DAYS = 10, 100, 1e-3, 16
WINSORIZE_LIMIT = 10.0
SEEDS = [0, 1, 2, 3, 4]

# (name, kwargs, graph_kind, shuffle_macro)
LADDER = [
    ("0_macro_only",      dict(macro_only=True),                              "none",   False),
    ("1_price_only",      dict(use_micro=False, use_graph=False, conditioning="none"), "none",   False),
    ("2_micro_no_graph",  dict(use_micro=True,  use_graph=False, conditioning="none"), "none",   False),
    ("3_graph_no_cond",   dict(use_micro=True,  use_graph=True,  conditioning="none"), "tweet",  False),
    ("4_full",            dict(use_micro=True,  use_graph=True,  conditioning="micro"), "tweet", False),
    ("5_shuffled_macro",  dict(use_micro=True,  use_graph=True,  conditioning="micro"), "tweet", True),
    ("6_attention_site",  dict(use_micro=True,  use_graph=True,  conditioning="attention"), "tweet", False),
    ("7_sector_graph",    dict(use_micro=True,  use_graph=True,  conditioning="micro"), "sector", False),
]


def build_everything():
    df = pd.read_parquet(EXACT)
    tickers = sorted(df["Ticker"].unique())
    days = sorted(pd.to_datetime(df["Target_Date"]).dt.strftime("%Y-%m-%d").unique())

    cache = EmbeddingCache.load(CACHE_PATH)
    if len(cache) == 0:
        print("empty embedding cache - run scripts/cache_experiment7_embeddings.py")
        sys.exit(1)

    macro_rows = pd.read_parquet("data/processed/fnspid_macro.parquet").to_dict("records")
    micro_rows = pd.read_parquet("data/processed/fnspid_micro.parquet").to_dict("records")
    calendar = load_macro_calendar(CALENDAR)
    tweets = load_ticker_tweets(FINAL)

    macro_vectors = build_macro_vectors(macro_rows, calendar, days, cache)
    micro_embeddings = build_micro_embeddings(tweets, micro_rows, tickers, days, cache)

    tweet_edges = {d: build_tweet_edges(micro_embeddings, tickers, d) for d in days}
    sector_graph = build_sector_graph(SECTOR_CSV)

    splits = tuple(build_daily_snapshots(df, s) for s in ("train", "dev", "test"))
    normalize_price_snapshots(*splits, winsorize_limit=WINSORIZE_LIMIT)
    return splits, macro_vectors, micro_embeddings, tweet_edges, sector_graph


def edges_for(kind: str, snap, tweet_edges: dict, sector_graph) -> torch.Tensor:
    if kind == "none":
        return torch.empty((2, 0), dtype=torch.long)
    if kind == "sector":
        return edge_index_for_tickers(sector_graph, snap.tickers)
    # Reuses Experiment 6's tested combine_edges rather than reimplementing the
    # local-index/both-directions logic. The "sector" key is just relation id 0
    # from RELATION_IDS -- these are tweet edges, and the model is
    # relation-agnostic (plain GATConv), so the id carries no meaning here.
    edge_index, _ = combine_edges({"sector": tweet_edges.get(snap.date, set())},
                                  snap.tickers)
    return edge_index


def micro_tensor(snap, micro_embeddings: dict) -> torch.Tensor:
    """Concatenates each ticker's tweet and news vectors into the 1536-d micro input.

    Concatenated rather than averaged so the model can weight news against tweets
    independently -- several ablation rows exist to measure exactly that.
    """
    rows = []
    for ticker in snap.tickers:
        entry = micro_embeddings.get((ticker, snap.date))
        if entry is None:
            rows.append(torch.zeros(1536))
        else:
            rows.append(torch.cat([entry[0], entry[1]]))
    return torch.stack(rows)


def run_epoch(model, snapshots, ctx, optimizer=None):
    macro_vectors, micro_embeddings, tweet_edges, sector_graph, kind, shuffled = ctx
    is_train = optimizer is not None
    model.train(is_train)
    order = list(snapshots)
    if is_train:
        random.shuffle(order)
        optimizer.zero_grad()

    probs_all, targets_all, days_all, total_loss = [], [], [], 0.0
    for i, snap in enumerate(order):
        macro = macro_vectors[shuffled[snap.date] if shuffled else snap.date]
        logits = model(snap.price_seq, snap.price_mask,
                       edges_for(kind, snap, tweet_edges, sector_graph),
                       micro_tensor(snap, micro_embeddings), macro)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            (loss / GRAD_ACCUM_DAYS).backward()
            if (i + 1) % GRAD_ACCUM_DAYS == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * len(snap.tickers)
        probs_all.append(torch.softmax(logits, dim=-1)[:, 1].detach())
        targets_all.append(snap.target)
        days_all.extend([snap.date] * len(snap.tickers))

    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy()
    metrics = pooled_metrics(targets, probs)
    metrics.update(within_day_metrics(targets, probs, days_all))
    return total_loss / len(targets), metrics


def train_one(row_name: str, kwargs: dict, kind: str, shuffle: bool, seed: int,
              splits: tuple, shared: tuple) -> dict:
    set_seed(seed)
    macro_vectors, micro_embeddings, tweet_edges, sector_graph = shared
    train_snaps, dev_snaps, test_snaps = splits

    shuffled = None
    if shuffle:
        days = sorted(macro_vectors)
        permuted = list(days)
        random.Random(seed).shuffle(permuted)
        shuffled = dict(zip(days, permuted))

    ctx = (macro_vectors, micro_embeddings, tweet_edges, sector_graph, kind, shuffled)
    model = MacroConditionedGraphNet(**kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_mcc, best_epoch, best_state, stale = -1.0, -1, None, 0
    for epoch in range(MAX_EPOCHS):
        run_epoch(model, train_snaps, ctx, optimizer)
        _, val = run_epoch(model, dev_snaps, ctx)
        if val["mcc"] > best_mcc:
            best_mcc, best_epoch, stale = val["mcc"], epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    _, test = run_epoch(model, test_snaps, ctx)
    test["n_samples"] = sum(len(s.tickers) for s in test_snaps)
    test["best_epoch"] = best_epoch
    test["best_val_mcc"] = best_mcc
    return test


def already_done(row_name: str, seed: int) -> bool:
    if not os.path.exists(RESULTS_CSV):
        return False
    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return False
    if "row" not in df.columns or "seed" not in df.columns:
        return False
    return bool(((df["row"] == row_name) & (df["seed"] == seed)).any())


def record(row_name: str, seed: int, test: dict) -> None:
    run_dir = os.path.join(OUT_DIR, f"{row_name}__seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump({"test": {"label": f"{row_name} seed={seed}", "backend": None, **test}},
                  fh, indent=2)
    row = {"row": row_name, "seed": seed, **test}
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a",
                               header=not os.path.exists(RESULTS_CSV), index=False)


def dump_film_by_day(model, snapshots, ctx) -> None:
    """Records ||gamma|| and ||beta|| per trading day for the explainability
    figure. Uses the trained full model on the test split."""
    macro_vectors, micro_embeddings, tweet_edges, sector_graph, kind, shuffled = ctx
    model.eval()
    rows = []
    with torch.no_grad():
        for snap in snapshots:
            model(snap.price_seq, snap.price_mask,
                  edges_for(kind, snap, tweet_edges, sector_graph),
                  micro_tensor(snap, micro_embeddings), macro_vectors[snap.date])
            if model.last_film is None:
                continue
            gamma, beta = model.last_film
            rows.append({"date": snap.date,
                         "gamma_norm": float(gamma.norm()),
                         "beta_norm": float(beta.norm())})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "film_by_day.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=None,
                    help="row names to run; default all")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("building embeddings, vectors and graphs...", flush=True)
    splits, macro_vectors, micro_embeddings, tweet_edges, sector_graph = build_everything()
    shared = (macro_vectors, micro_embeddings, tweet_edges, sector_graph)
    density = [len(v) for v in tweet_edges.values()]
    print(f"tweet graph: mean {sum(density)/max(len(density),1):.1f} edges/day, "
          f"max {max(density) if density else 0}")

    for row_name, kwargs, kind, shuffle in LADDER:
        if args.rows and row_name not in args.rows:
            continue
        for seed in SEEDS:
            if already_done(row_name, seed):
                print(f"  skip  {row_name} seed={seed}")
                continue
            print(f"  train {row_name} seed={seed} ...", flush=True)
            test = train_one(row_name, kwargs, kind, shuffle, seed, splits, shared)
            record(row_name, seed, test)
            print(f"        pooled MCC={test['mcc']:+.4f} AUC={test['auc']:.4f} | "
                  f"within-day MCC={test['within_day_mcc']:+.4f} "
                  f"({test['days_scored']} days)")
            if test["mcc"] > 0.15:
                print("        *** TRIPWIRE: pooled MCC > 0.15 - audit for leakage "
                      "before treating this as a result ***")

    # Dump FiLM explainability for the best seed of row 4_full
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            if "4_full" in df["row"].values:
                best_row = df[df["row"] == "4_full"].nlargest(1, "mcc").iloc[0]
                best_seed = int(best_row["seed"])
                print(f"\nExtracting FiLM for 4_full seed={best_seed} ...", flush=True)
                _, kwargs, kind, shuffle = LADDER[4]  # 4_full is always at index 4
                _, _, test_snaps = splits
                # train_one returns metrics, not the model, so the model is
                # rebuilt here with the same seed to recover its FiLM parameters.
                set_seed(best_seed)
                model = MacroConditionedGraphNet(**kwargs)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

                shuffled = None
                ctx = (macro_vectors, micro_embeddings, tweet_edges, sector_graph, kind, shuffled)
                best_mcc, best_epoch, best_state, stale = -1.0, -1, None, 0
                train_snaps, dev_snaps, _ = splits
                for epoch in range(MAX_EPOCHS):
                    run_epoch(model, train_snaps, ctx, optimizer)
                    _, val = run_epoch(model, dev_snaps, ctx)
                    if val["mcc"] > best_mcc:
                        best_mcc, best_epoch, stale = val["mcc"], epoch, 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        stale += 1
                    if stale >= PATIENCE:
                        break

                model.load_state_dict(best_state)
                dump_film_by_day(model, test_snaps, ctx)
                print(f"FiLM dump written to {os.path.join(OUT_DIR, 'film_by_day.csv')}")
        except Exception as e:
            print(f"Could not extract FiLM: {e}")

    print(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
