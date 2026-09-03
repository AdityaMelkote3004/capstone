"""Precomputes per-trading-day edge sets for Experiment 6 and caches them.

The correlation builder runs an 87x87 correlation per trading day; without
this cache the 84-run grid would repeat that work 84 times. The cache is
also what lets a crashed Colab session resume rather than restart.
"""

import pickle

import pandas as pd

from src.data.graph_edges import (build_comention_edges, build_correlation_edges,
                                  build_sector_edges, load_event_texts,
                                  load_returns_matrix)


def build_edge_cache(exact_parquet: str, final_parquet: str, sector_csv: str,
                     out_path: str, window: int = 60, k: int = 5,
                     lookback: int = 5) -> dict:
    """Builds {date_string: {relation: set_of_pairs}} for every trading day in
    the exact-protocol dataset, and pickles it to `out_path`."""
    exact = pd.read_parquet(exact_parquet, columns=["Target_Date"])
    dates = sorted(pd.to_datetime(exact["Target_Date"]).unique())

    returns = load_returns_matrix(final_parquet)
    texts = load_event_texts(final_parquet)
    universe = set(returns.columns)
    sector = build_sector_edges(sector_csv)

    cache = {}
    for date in dates:
        ts = pd.Timestamp(date)
        cache[ts.strftime("%Y-%m-%d")] = {
            "sector": sector,
            "correlation": build_correlation_edges(returns, ts, window=window, k=k),
            "comention": build_comention_edges(texts, ts, universe, lookback=lookback),
        }

    with open(out_path, "wb") as fh:
        pickle.dump(cache, fh)
    return cache


def load_edge_cache(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
