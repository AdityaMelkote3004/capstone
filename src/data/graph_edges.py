"""Leakage-safe edge builders for Experiment 6's dense stock graph.

Every builder takes an explicit time cutoff and returns undirected
`(ticker_a, ticker_b)` pairs for one target date, where each pair is a
single alphabetically sorted tuple. No builder may use data from the
target date or later -- see
docs/superpowers/specs/2026-09-03-dense-graph-experiment-design.md and the
leakage tests in tests/test_graph_edges.py.
"""

import pandas as pd


def load_returns_matrix(final_parquet_path: str) -> pd.DataFrame:
    """Loads a (date x ticker) matrix of daily returns from the corrected
    modeling set. The matrix is ~39% NaN: all 87 tickers have gaps inside
    their own active range, which is why correlation uses a 60-day window
    and a min_periods guard."""
    df = pd.read_parquet(final_parquet_path, columns=["Date", "Ticker", "Return"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot_table(index="Date", columns="Ticker", values="Return").sort_index()


def build_correlation_edges(returns: pd.DataFrame, target_date, window: int = 60,
                            k: int = 5, min_periods: int = 10) -> set:
    """Connects each ticker to its top-`k` most correlated partners by |rho|,
    computed over the `window` trading days STRICTLY BEFORE `target_date`.

    Top-k rather than a |rho| threshold: a 0.5 threshold measured 2130 edges
    (57% density), which would rebuild the over-smoothing collapse diagnosed
    in Experiment 4. Top-k makes density a controlled parameter.

    Returns an empty set when there is insufficient history (early dates),
    rather than raising.
    """
    idx = returns.index.searchsorted(pd.Timestamp(target_date))
    win = returns.iloc[max(0, idx - window):idx]   # ends at t-1; excludes t
    if len(win) < min_periods:
        return set()

    corr = win.corr(min_periods=min_periods).abs()
    for ticker in corr.columns:
        corr.at[ticker, ticker] = float("nan")     # never self-edge

    edges = set()
    for ticker in corr.columns:
        partners = corr[ticker].dropna().sort_values(ascending=False)
        for partner in partners.head(k).index:
            edges.add(tuple(sorted((ticker, partner))))
    return edges
