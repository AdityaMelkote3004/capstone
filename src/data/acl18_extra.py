"""Experiment 10's static node features, aligned to the ACL18 (day, stock) grid.

Exp 10 found that putting volatility, EDGAR fundamentals, tweet counts and the
peer/sector block on each stock raised within-day direction MCC from about
+0.009 (price only) to about +0.046 -- the largest stock-discrimination gain in
the project. This module hands the same features to DTML as an additive context.

Every feature is computed from information strictly before the target date
(see src/data/node_features.py); standardisation uses train-split rows only.
Tweet-text PCA is deliberately left out: Exp 8 showed its apparent signal was
missingness encoding, which the tweet-count features already carry.
"""

import numpy as np
import pandas as pd

from src.data.edge_cache import load_edge_cache
from src.data.node_features import (assemble, build_peer_features,
                                    build_volatility_features, standardise)

FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
EDGE_CACHE = "results/experiment6_dense_graph/edge_cache.pkl"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
COUNT_SOURCE = "dataset/stocknet_protocol_a_exact.parquet"


def extra_features(data: dict, peer: bool = True) -> np.ndarray:
    """Leak-free profile (v2). See extra_features_v2."""
    return extra_features_v2(data, peer=peer)


# ---------------------------------------------------------------------------
# v2: strictly-prior profile.
#
# The v1 builder below keyed every feature to the TARGET date in the modeling
# parquet, which only holds labelled rows. A row exists at (stock, t) exactly
# when that stock's move on t left StockNet's buffer zone (checked: 26,584 /
# 16,782 cells, zero exceptions). Row presence on t is therefore target-day
# information, and it reached the model through the peer neighbour count
# (corr 0.947 with the number of neighbours labelled at t), the peer means and
# the zero-filled profiles of unlabelled stocks. v2 uses only rows dated
# strictly before t; tests/test_profile_leakage.py verifies this by rebuilding
# the profile with all data from t onward deleted and requiring equality.
# ---------------------------------------------------------------------------

EWMA_LAMBDA = 0.94
FUND_COLS = ["Revenue", "NetIncome", "TotalAssets", "TotalLiabilities",
             "StockholdersEquity", "EPS", "Cash", "Revenue_Growth",
             "NetIncome_Growth", "EPS_Growth", "Profit_Margin", "Debt_To_Equity", "ROA"]
FUND_MISSING = ["Revenue_Missing", "NetIncome_Missing", "TotalAssets_Missing",
                "TotalLiabilities_Missing", "StockholdersEquity_Missing",
                "EPS_Missing", "Cash_Missing"]
LOG_LEVELS = ["Revenue", "NetIncome", "TotalAssets", "TotalLiabilities",
              "StockholdersEquity", "Cash"]


def _acl18_returns(tickers, root="data/acl18"):
    import os
    r = np.stack([np.genfromtxt(os.path.join(root, "ourpped", f"{t}.csv"),
                                delimiter=",")[:, 3] for t in tickers], axis=1)
    r[r < -1e5] = np.nan
    days = list(np.genfromtxt(os.path.join(root, "trading_dates.csv"), dtype=str))
    return pd.DataFrame(r, index=days, columns=tickers)


def build_profile_v2(tickers, target_days, final: pd.DataFrame, returns: pd.DataFrame,
                     edge_cache: dict, sector_csv: str, peer: bool = True):
    """(len(target_days), len(tickers), F). Every value uses data dated < t only.

    `final` and `returns` may be truncated by the caller (leakage test); nothing
    here reads a row dated on or after the target day."""
    cal = list(returns.index)
    pos = {d: i for i, d in enumerate(cal)}
    R = returns.to_numpy()
    S = len(tickers)

    # volatility block from the complete ACL18 return calendar
    vol = np.full((len(target_days), S, 6), np.nan, dtype=np.float64)
    for k, t in enumerate(target_days):
        i = pos[t]
        past = R[:i]                                   # rows strictly before t
        for j, w in enumerate((5, 10, 20)):
            blk = past[-w:]
            vol[k, :, j] = np.nanstd(blk, axis=0, ddof=1) if len(blk) > 1 else np.nan
        if len(past):
            wts = (1 - EWMA_LAMBDA) * EWMA_LAMBDA ** np.arange(len(past))[::-1]
            sq = np.where(np.isnan(past), 0.0, past ** 2)
            norm = (np.where(np.isnan(past), 0.0, 1.0) * wts[:, None]).sum(0)
            vol[k, :, 3] = np.sqrt((sq * wts[:, None]).sum(0) / np.maximum(norm, 1e-12))
            vol[k, :, 4] = past[-1]
            vol[k, :, 5] = np.abs(past[-1])

    # fundamentals and tweet counts from modeling-set rows strictly before t
    f = final.copy()
    f["day"] = pd.to_datetime(f["Date"]).dt.strftime("%Y-%m-%d")
    for c in LOG_LEVELS:
        f[c] = np.sign(f[c]) * np.log1p(f[c].abs())
    f = f.sort_values("day")
    fund = np.full((len(target_days), S, len(FUND_COLS) + len(FUND_MISSING)), np.nan)
    tw = np.zeros((len(target_days), S, 5))
    tweet_by = {(r.Ticker, r.day): r.Total_Tweet_Count for r in
                f[["Ticker", "day", "Total_Tweet_Count"]].itertuples()}
    by_ticker = {t: g for t, g in f.groupby("Ticker")}
    for s, tk in enumerate(tickers):
        g = by_ticker.get(tk)
        if g is None:
            continue
        gdays = g["day"].to_numpy()
        vals = g[FUND_COLS + FUND_MISSING].to_numpy(dtype=np.float64)
        for k, t in enumerate(target_days):
            j = np.searchsorted(gdays, t, side="left") - 1   # last row dated < t
            if j >= 0:
                fund[k, s] = vals[j]
            i = pos[t]
            counts = [tweet_by.get((tk, cal[i - q]), 0.0) if i - q >= 0 else 0.0
                      for q in range(1, 5)]
            tw[k, s, :4] = np.log1p(np.clip(counts, 0, None))
            tw[k, s, 4] = np.log1p(np.clip(sum(counts), 0, None))

    blocks = [vol, fund, tw]
    if peer:
        sectors = pd.read_csv(sector_csv)
        sec_of = dict(zip(sectors["Ticker"], sectors["Sector"]))
        names = sorted(set(sec_of.values()))
        idx = {t: i for i, t in enumerate(tickers)}
        src = vol[..., [4, 5, 3]]                      # prior-day return, |return|, EWMA
        pe = np.zeros((len(target_days), S, 7 + len(names)))
        for k, t in enumerate(target_days):
            adj = {}
            for rel in edge_cache.get(t, {}).values():
                for a, b in rel:
                    if a in idx and b in idx:
                        adj.setdefault(idx[a], set()).add(idx[b])
                        adj.setdefault(idx[b], set()).add(idx[a])
            for s in range(S):
                nb = sorted(adj.get(s, ()))
                if nb:
                    m = np.nanmean(src[k, nb], axis=0)
                    pe[k, s, 0:3] = m
                    pe[k, s, 3:6] = src[k, s] - m
                    pe[k, s, 6] = np.log1p(len(nb))     # graph degree, prior info only
                pe[k, s, 7 + names.index(sec_of[tickers[s]])] = 1.0
        blocks.append(pe)
    out = np.concatenate(blocks, axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extra_features_v2(data: dict, peer: bool = True) -> np.ndarray:
    cols = ["Date", "Ticker", "Total_Tweet_Count"] + FUND_COLS + FUND_MISSING
    final = pd.read_parquet(FINAL, columns=cols)
    X = build_profile_v2(data["tickers"], data["dates"], final,
                         _acl18_returns(data["tickers"]), load_edge_cache(EDGE_CACHE),
                         SECTOR_CSV, peer=peer)
    train = data["split"] == "train"
    flat = X[train].reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    return np.clip((X - mu) / sd, -10, 10).astype(np.float32)


def extra_features_v1_leaky(data: dict, peer: bool = True) -> np.ndarray:
    """v1, kept only for audit. Leaks target-day buffer-zone membership."""
    tickers, dates = data["tickers"], data["dates"]
    grid = pd.DataFrame([(t, d) for d in dates for t in tickers],
                        columns=["Ticker", "Target_Date"])

    # Tweet counts live in the exact-protocol table; rows absent there (buffer
    # days, unlabelled stocks) get zero counts, which is what "no tweets" means.
    counts = pd.read_parquet(COUNT_SOURCE)
    count_cols = [c for c in counts.columns if c.endswith("_TweetCount")]
    counts["Target_Date"] = pd.to_datetime(counts["Target_Date"]).dt.strftime("%Y-%m-%d")
    grid = grid.merge(counts[["Ticker", "Target_Date"] + count_cols],
                      on=["Ticker", "Target_Date"], how="left")
    grid[count_cols] = grid[count_cols].fillna(0.0)

    matrix, _ = assemble(grid, FINAL, text_pca=None)
    if peer:
        peer_m, _ = build_peer_features(grid, build_volatility_features(FINAL),
                                        load_edge_cache(EDGE_CACHE), SECTOR_CSV)
        matrix = np.concatenate([matrix, peer_m], axis=1)

    is_train = np.repeat(data["split"] == "train", len(tickers))
    matrix = standardise(matrix, is_train)
    return matrix.reshape(len(dates), len(tickers), -1).astype(np.float32)
