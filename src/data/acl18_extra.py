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
    """Returns (days, stocks, F) float32 aligned to load_acl18()'s grid."""
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
