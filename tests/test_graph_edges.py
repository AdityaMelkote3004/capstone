import numpy as np
import pandas as pd
import pytest

from src.data.graph_edges import build_correlation_edges, load_returns_matrix


def _returns_frame():
    """5 tickers x 80 trading days. AAA/BBB move together, CCC/DDD move
    together, EEE is pure noise on its own schedule."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=80)
    base1 = rng.normal(0, 1, 80)
    base2 = rng.normal(0, 1, 80)
    return pd.DataFrame(
        {
            "AAA": base1 + rng.normal(0, 0.01, 80),
            "BBB": base1 + rng.normal(0, 0.01, 80),
            "CCC": base2 + rng.normal(0, 0.01, 80),
            "DDD": base2 + rng.normal(0, 0.01, 80),
            "EEE": rng.normal(0, 1, 80),
        },
        index=dates,
    )


def test_finds_the_correlated_pairs():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[70], window=60, k=1)
    assert ("AAA", "BBB") in edges
    assert ("CCC", "DDD") in edges


def test_edges_are_sorted_tuples_without_duplicates():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[70], window=60, k=2)
    for a, b in edges:
        assert a < b, "each edge must be a single alphabetically sorted tuple"
    assert len(edges) == len(set(edges))


def test_no_self_edges():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[70], window=60, k=3)
    assert all(a != b for a, b in edges)


def test_top_k_caps_degree():
    df = _returns_frame()
    k = 2
    edges = build_correlation_edges(df, df.index[70], window=60, k=k)
    # Each node proposes at most k partners; the union is undirected, so the
    # total can never exceed k * n_tickers (and is usually far less).
    assert len(edges) <= k * df.shape[1]


def test_leakage_target_day_cannot_change_edges():
    """THE leakage test: corrupting the target day and every day after it
    must not change the edges produced for that target day."""
    df = _returns_frame()
    target = df.index[70]
    clean = build_correlation_edges(df, target, window=60, k=2)

    poisoned = df.copy()
    poisoned.loc[target:] = 999.0   # target day onward is garbage
    after = build_correlation_edges(poisoned, target, window=60, k=2)

    assert clean == after, "edges for day t must not depend on data from t onward"


def test_insufficient_history_returns_empty_not_error():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[2], window=60, k=5, min_periods=10)
    assert edges == set()


def test_min_periods_rejects_thin_overlap():
    """Two tickers overlapping on only a handful of days must not produce an
    edge, however perfectly those few days correlate."""
    dates = pd.bdate_range("2020-01-01", periods=80)
    df = pd.DataFrame(
        {
            "AAA": np.arange(80, dtype=float),
            "BBB": np.arange(80, dtype=float),
            "CCC": np.nan,
        },
        index=dates,
    )
    # CCC overlaps AAA on only 3 days, perfectly correlated on those days.
    df.iloc[60:63, df.columns.get_loc("CCC")] = [1.0, 2.0, 3.0]
    edges = build_correlation_edges(df, df.index[70], window=60, k=4, min_periods=10)
    assert ("AAA", "CCC") not in edges
    assert ("BBB", "CCC") not in edges


def test_load_returns_matrix_shape_on_real_data():
    df = load_returns_matrix("dataset/final/stocknet_final_modeling_set.parquet")
    assert df.shape == (504, 87)
    assert df.notna().sum().sum() == 26603
