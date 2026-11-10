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
    print("PASS: profile uses only data dated before the target day"
          if worst == 0.0 else f"FAIL: max difference {worst}")
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
