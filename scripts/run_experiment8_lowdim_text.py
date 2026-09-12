"""Experiment 8 -- does LESS text beat MORE text?

Experiments 1-7 produced one consistent, counter-intuitive result: tweet COUNTS
beat mean-pooled FinBERT semantics on the same test split, across four
architectures (+0.092 vs +0.036 pooled MCC). Experiment 8 tests the explanation
directly by varying only the text representation's DIMENSIONALITY, holding the
model, split, seeds and metrics fixed.

Rows, in increasing text richness:

  A_price            12 price features, no text            (control floor)
  B_price_counts     + 4 tweet counts                      (Exp 1's winner, now multi-seed)
  C_counts_pca8      + 8 PCA dims of FinBERT
  D_counts_pca16     + 16 PCA dims of FinBERT
  E_pca16_only       PCA text but NO counts                (isolates semantics)
  F_counts_full768   + the raw 768-d mean-pooled vector    (reproduces the failure)

If the "dimensionality, not content" explanation is right, C/D beat F, and B is
already close to the best. If richer text genuinely helps, F wins. Either way
the comparison is apples-to-apples for the first time -- previously the 768-d
and count-based results came from different experiments with different models.

LEAKAGE: PCA is fit on TRAIN-split rows only, then applied to dev and test.
Fitting it on all rows would let test-set covariance choose the axes.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.lowdim_text import LowDimTextLSTM
from src.utils.metrics import pooled_metrics, within_day_metrics
from src.utils.seed import set_seed

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
TEXT_CACHE = "data/processed/experiment8_text_matrix.pt"
OUT_DIR = "results/experiment8_lowdim_text"
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")

PRICE_COLS = [f"Day{d}_Price{p}" for d in range(1, 5) for p in range(1, 4)]
COUNT_COLS = [f"Day{d}_TweetCount" for d in range(1, 5)]

EPOCHS, PATIENCE, LR, BATCH = 100, 15, 1e-3, 128
MIN_EPOCHS = 10
SEEDS = [0, 1, 2, 3, 4]
WINSORIZE = 10.0

ROWS = {
    "A_price":          dict(counts=False, text=None),
    "B_price_counts":   dict(counts=True,  text=None),
    "C_counts_pca8":    dict(counts=True,  text=8),
    "D_counts_pca16":   dict(counts=True,  text=16),
    "E_pca16_only":     dict(counts=False, text=16),
    "F_counts_full768": dict(counts=True,  text=768),
    # Experiment 1 reported +0.0922 pooled MCC from ONE run of an LSTM with
    # hidden=64, 2 layers, dropout=0.2. This row reproduces that architecture
    # exactly, at 5 seeds, to test whether that number was a lucky draw.
    "G_exp1arch_counts": dict(counts=True, text=None,
                              arch=dict(hidden=64, layers=2, dropout=0.2)),
    # Maps the dimensionality curve. If "compression is what matters" is right,
    # pooled MCC should rise as dims fall and then fall again once the
    # representation is too small to carry anything.
    "H_counts_pca2":    dict(counts=True, text=2),
    "I_counts_pca4":    dict(counts=True, text=4),
    "J_counts_pca32":   dict(counts=True, text=32),
    "K_counts_pca64":   dict(counts=True, text=64),
}


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(EXACT)
    df["day"] = pd.to_datetime(df["Target_Date"]).dt.strftime("%Y-%m-%d")
    return df


def build_text_matrix(df: pd.DataFrame) -> torch.Tensor:
    """Returns a (n_rows, 768) tensor of tweet embeddings aligned to `df`'s order.

    Cached, because building it walks the whole FinBERT cache and takes minutes.
    """
    if os.path.exists(TEXT_CACHE):
        cached = torch.load(TEXT_CACHE, weights_only=False)
        if cached["n"] == len(df):
            print(f"  text matrix from cache ({cached['n']} rows)")
            return cached["matrix"]

    print("  building text matrix (walking the FinBERT cache)...", flush=True)
    from src.data.micro_encoder import build_micro_embeddings, load_ticker_tweets
    from src.models.finbert_cache import EmbeddingCache

    cache = EmbeddingCache.load("data/processed/experiment7_embeddings.pt")
    micro_rows = pd.read_parquet("data/processed/fnspid_micro.parquet").to_dict("records")
    tweets = load_ticker_tweets("dataset/final/stocknet_final_modeling_set.parquet")
    tickers = sorted(df["Ticker"].unique())
    days = sorted(df["day"].unique())
    micro = build_micro_embeddings(tweets, micro_rows, tickers, days, cache,
                                   tweet_lookback=20)
    rows = [micro.get((t, d), (torch.zeros(768), None))[0]
            for t, d in zip(df["Ticker"], df["day"])]
    matrix = torch.stack(rows)
    os.makedirs(os.path.dirname(TEXT_CACHE), exist_ok=True)
    torch.save({"n": len(df), "matrix": matrix}, TEXT_CACHE)
    return matrix


def fit_pca(train_matrix: np.ndarray, n_components: int):
    """PCA fit on TRAIN rows only. Returns (mean, components) for projection."""
    mean = train_matrix.mean(axis=0)
    centred = train_matrix - mean
    # SVD rather than covariance eigendecomposition: 768 dims, 20k rows, and
    # randomized SVD keeps this to a second or two.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return mean, vt[:n_components]


def prepare(df: pd.DataFrame, text_matrix: torch.Tensor, spec: dict):
    """Builds (per-day tensor, lengths, text tensor, targets, days) per split."""
    per_day = []
    for d in range(1, 5):
        cols = [f"Day{d}_Price{p}" for p in range(1, 4)]
        if spec["counts"]:
            cols = cols + [f"Day{d}_TweetCount"]
        per_day.append(df[cols].to_numpy(dtype=np.float32))
    seq = np.stack(per_day, axis=1)                       # (n, 4, per_day_dim)

    is_train = (df["Split"] == "train").to_numpy()
    # Winsorize before standardising -- Price3 spans [-6495, +6500] because of
    # BRK-A/PCLN and would otherwise crush every other ticker to ~0 (see
    # src/data/price_fix.py). Counts get log1p for the same reason.
    seq = np.clip(seq, -WINSORIZE, WINSORIZE)
    if spec["counts"]:
        seq[:, :, 3] = np.log1p(np.clip(seq[:, :, 3], 0, None))
    flat = seq.reshape(-1, seq.shape[-1])
    train_flat = seq[is_train].reshape(-1, seq.shape[-1])
    mu, sd = train_flat.mean(axis=0), train_flat.std(axis=0) + 1e-8
    seq = ((flat - mu) / sd).reshape(seq.shape)

    text = None
    if spec["text"]:
        matrix = text_matrix.numpy()
        if spec["text"] == 768:
            projected = matrix
        else:
            mean, components = fit_pca(matrix[is_train], spec["text"])
            projected = (matrix - mean) @ components.T
        tmu = projected[is_train].mean(axis=0)
        tsd = projected[is_train].std(axis=0) + 1e-8
        text = torch.tensor((projected - tmu) / tsd, dtype=torch.float32)

    return (torch.tensor(seq, dtype=torch.float32),
            torch.tensor(df["Window_Length"].to_numpy(), dtype=torch.long),
            text,
            torch.tensor(df["Target"].to_numpy(), dtype=torch.long),
            df["day"].to_numpy())


def evaluate(model, seq, lengths, text, targets, days, idx):
    model.eval()
    with torch.no_grad():
        logits = model(seq[idx], lengths[idx], None if text is None else text[idx])
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    y = targets[idx].numpy()
    metrics = pooled_metrics(y, probs)
    metrics.update(within_day_metrics(y, probs, days[idx]))
    # Guards the protocol-A invariant: test must stay at 3720 samples, or these
    # numbers are not comparable with Experiments 1 and 4-7.
    metrics["n_samples"] = int(len(y))
    return metrics


def train_one(row: str, spec: dict, seed: int, data) -> dict:
    seq, lengths, text, targets, days, splits = data
    set_seed(seed)
    tr = np.where(splits == "train")[0]
    dv = np.where(splits == "dev")[0]
    te = np.where(splits == "test")[0]

    model = LowDimTextLSTM(per_day_dim=seq.shape[-1],
                           text_dim=0 if text is None else text.shape[1],
                           **spec.get("arch", {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_mcc, best_state, stale = -2.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.permutation(tr)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            optimizer.zero_grad()
            logits = model(seq[batch], lengths[batch],
                           None if text is None else text[batch])
            nn.functional.cross_entropy(logits, targets[batch]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val = evaluate(model, seq, lengths, text, targets, days, dv)
        if val["mcc"] > best_mcc:
            best_mcc, stale = val["mcc"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test = evaluate(model, seq, lengths, text, targets, days, te)
    test["best_val_mcc"] = best_mcc
    return test


def record(row: str, seed: int, metrics: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    entry = {"row": row, "seed": seed, **metrics}
    frame = pd.DataFrame([entry])
    frame.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV),
                 index=False)


def already_done(row: str, seed: int) -> bool:
    if not os.path.exists(RESULTS_CSV):
        return False
    d = pd.read_csv(RESULTS_CSV)
    return ((d.row == row) & (d.seed == seed)).any()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--out", default=None,
                    help="alternate results CSV, so two runs can proceed "
                         "concurrently without interleaving appends")
    args = ap.parse_args()
    if args.out:
        global RESULTS_CSV
        RESULTS_CSV = args.out

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_frame()
    print(f"protocol A: {len(df)} rows, test n={(df.Split=='test').sum()}")

    needs_text = any(ROWS[r]["text"] for r in (args.rows or ROWS))
    text_matrix = build_text_matrix(df) if needs_text else torch.zeros(len(df), 768)

    for row, spec in ROWS.items():
        if args.rows and row not in args.rows:
            continue
        seq, lengths, text, targets, days = prepare(df, text_matrix, spec)
        data = (seq, lengths, text, targets, days, df["Split"].to_numpy())
        dim = seq.shape[-1] * 4 + (0 if text is None else text.shape[1])
        print(f"\n{row}  ({dim} total input features)")
        for seed in SEEDS:
            if already_done(row, seed):
                print(f"  skip  seed={seed}")
                continue
            test = train_one(row, spec, seed, data)
            record(row, seed, test)
            print(f"  seed={seed}  pooled MCC={test['mcc']:+.4f}  "
                  f"within-day={test['within_day_mcc']:+.4f}  "
                  f"acc={test['accuracy']:.4f}  n={test['n_samples']}", flush=True)

    print(f"\nDone. {RESULTS_CSV}")


if __name__ == "__main__":
    main()
