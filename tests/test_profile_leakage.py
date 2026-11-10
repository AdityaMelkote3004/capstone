"""Truncation test for the stock profile used by MMGT.

For a sample of target days t, the profile is built twice: once from the full
data and once from data truncated so that nothing dated on or after t exists.
Any difference means the profile reads information from day t or later.
The v1 (leaky) builder fails this test; v2 must pass it exactly.

Run: python tests/test_profile_leakage.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.acl18 import load_acl18
from src.data.acl18_extra import (EDGE_CACHE, FINAL, FUND_COLS, FUND_MISSING, SECTOR_CSV,
                                  _acl18_returns, build_profile_v2)
from src.data.edge_cache import load_edge_cache


def main():
    data = load_acl18(window=10)
    tickers = data["tickers"]
    final = pd.read_parquet(FINAL, columns=["Date", "Ticker", "Total_Tweet_Count"]
                            + FUND_COLS + FUND_MISSING)
    final_days = pd.to_datetime(final["Date"]).dt.strftime("%Y-%m-%d")
    returns = _acl18_returns(tickers)
    cache = load_edge_cache(EDGE_CACHE)

    split = data["split"]
    rng = np.random.default_rng(0)
    sample = []
    for s in ("train", "val", "test"):
        days = [d for d, sp in zip(data["dates"], split) if sp == s]
        sample += list(rng.choice(days, 4, replace=False))

    worst = 0.0
    for t in sample:
        full = build_profile_v2(tickers, [t], final, returns, cache, SECTOR_CSV)
        cut = build_profile_v2(tickers, [t], final[final_days < t],
                               returns[returns.index < t].reindex(returns.index),
                               cache, SECTOR_CSV)
        diff = np.abs(full - cut).max()
        worst = max(worst, diff)
        print(f"  {t}: max |full - truncated| = {diff:.3g}")
    # The graph edges feeding the peer features: rebuild correlation and
    # co-mention edges from data truncated before t and compare with the cache.
    from src.data.graph_edges import (build_comention_edges, build_correlation_edges,
                                      load_event_texts, load_returns_matrix)
    rets = load_returns_matrix(FINAL)
    texts = load_event_texts(FINAL)
    universe = set(rets.columns)
    edge_mismatch = 0
    for t in sample:
        t = str(t)
        ts = pd.Timestamp(t)
        corr = build_correlation_edges(rets[rets.index < ts], ts)
        com = build_comention_edges({d: v for d, v in texts.items() if d < ts}, ts, universe)
        ok = (corr == set(cache[t]["correlation"])) and (com == set(cache[t]["comention"]))
        edge_mismatch += 0 if ok else 1
        print(f"  {t}: edges rebuilt from truncated data match cache: {ok}")

    passed = worst == 0.0 and edge_mismatch == 0
    print("PASS: profile and graph use only data dated before the target day"
          if passed else f"FAIL: profile diff {worst}, edge mismatches {edge_mismatch}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
