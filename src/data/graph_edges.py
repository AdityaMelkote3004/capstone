"""Leakage-safe edge builders for Experiment 6's dense stock graph.

Every builder takes an explicit time cutoff and returns undirected
`(ticker_a, ticker_b)` pairs for one target date, where each pair is a
single alphabetically sorted tuple. No builder may use data from the
target date or later -- see
docs/superpowers/specs/2026-09-03-dense-graph-experiment-design.md and the
leakage tests in tests/test_graph_edges.py.
"""

import re

import pandas as pd

_CASHTAG = re.compile(r"\$\s*([A-Za-z]{1,5})")


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


def extract_cashtags(text: str, universe: set) -> set:
    """Pulls universe tickers out of a tweet. The corpus writes cashtags with
    a space after the dollar sign (`$ aapl`), hence the `\\s*`."""
    found = {m.upper() for m in _CASHTAG.findall(str(text))}
    return found & universe


def load_event_texts(final_parquet_path: str) -> dict:
    """Maps each date to the list of that date's event-tweet texts, pooled
    across tickers. Rows whose Event_Texts is empty are skipped."""
    df = pd.read_parquet(final_parquet_path, columns=["Date", "Event_Texts"])
    df["Date"] = pd.to_datetime(df["Date"])
    out: dict = {}
    for date, text in zip(df["Date"], df["Event_Texts"]):
        if text is None or len(str(text)) <= 2:
            continue
        out.setdefault(date, []).extend(str(text).split("[SEP]"))
    return out


def build_comention_edges(texts_by_date: dict, target_date, universe: set,
                          lookback: int = 5) -> set:
    """Connects tickers that appear together in the same tweet, over the
    `lookback` days STRICTLY BEFORE `target_date`.

    Measured on the real corpus: 1785 tweet segments mention 2+ universe
    tickers, spread across 347 of 504 trading days.
    """
    target = pd.Timestamp(target_date)
    window_start = target - pd.Timedelta(days=lookback)

    edges = set()
    for date, texts in texts_by_date.items():
        if not (window_start <= date < target):    # excludes the target day
            continue
        for text in texts:
            mentioned = sorted(extract_cashtags(text, universe))
            for i in range(len(mentioned)):
                for j in range(i + 1, len(mentioned)):
                    edges.add((mentioned[i], mentioned[j]))
    return edges
