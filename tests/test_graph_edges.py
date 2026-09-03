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


from src.data.graph_edges import (build_comention_edges, extract_cashtags,
                                  load_event_texts)

UNIVERSE = {"AAPL", "MSFT", "GOOG", "BAC"}


def test_extract_cashtags_handles_the_corpus_space_format():
    # The corpus writes cashtags as "$ aapl", with a space after the dollar.
    got = extract_cashtags("$ aapl - online retail sales up -> URL $ msft", UNIVERSE)
    assert got == {"AAPL", "MSFT"}


def test_extract_cashtags_ignores_tickers_outside_the_universe():
    got = extract_cashtags("$ aapl and $ tsla and $ zzzz", UNIVERSE)
    assert got == {"AAPL"}


def test_comention_pairs_every_mentioned_ticker():
    texts = {pd.Timestamp("2020-01-06"): ["$ aapl beats $ msft and $ goog today"]}
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-07"), UNIVERSE, lookback=5)
    assert edges == {("AAPL", "GOOG"), ("AAPL", "MSFT"), ("GOOG", "MSFT")}


def test_single_ticker_mention_creates_no_edge():
    texts = {pd.Timestamp("2020-01-06"): ["$ aapl going up"]}
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-07"), UNIVERSE, lookback=5)
    assert edges == set()


def test_leakage_target_day_tweets_are_excluded():
    """Tweets dated on the target day itself must never inform that day."""
    texts = {
        pd.Timestamp("2020-01-06"): ["$ aapl with $ msft"],
        pd.Timestamp("2020-01-07"): ["$ goog with $ bac"],   # the target day
    }
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-07"), UNIVERSE, lookback=5)
    assert ("AAPL", "MSFT") in edges
    assert ("BAC", "GOOG") not in edges, "target-day tweets must be excluded"


def test_lookback_window_is_respected():
    texts = {
        pd.Timestamp("2020-01-01"): ["$ aapl with $ msft"],   # far in the past
        pd.Timestamp("2020-01-20"): ["$ goog with $ bac"],    # just before target
    }
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-21"), UNIVERSE, lookback=3)
    assert ("BAC", "GOOG") in edges
    assert ("AAPL", "MSFT") not in edges, "outside the lookback window"


def test_load_event_texts_on_real_data():
    texts = load_event_texts("dataset/final/stocknet_final_modeling_set.parquet")
    assert len(texts) > 300
    assert all(isinstance(v, list) for v in texts.values())


import torch

from src.data.graph_edges import build_sector_edges, combine_edges

RELATION = {"sector": 0, "correlation": 1, "comention": 2}


def test_build_sector_edges_matches_experiment_4_graph():
    edges = build_sector_edges("dataset/final/sector_mapping.csv")
    assert len(edges) == 381
    assert all(a < b for a, b in edges)


def test_combine_edges_uses_local_indices_and_both_directions():
    tickers = ["AAA", "BBB", "CCC"]
    ei, et = combine_edges({"sector": {("AAA", "BBB")}}, tickers)
    pairs = set(map(tuple, ei.t().tolist()))
    assert pairs == {(0, 1), (1, 0)}
    assert et.tolist() == [0, 0]


def test_combine_edges_tags_each_relation():
    tickers = ["AAA", "BBB", "CCC"]
    ei, et = combine_edges(
        {"sector": {("AAA", "BBB")}, "comention": {("BBB", "CCC")}}, tickers
    )
    by_pair = {tuple(p): t for p, t in zip(ei.t().tolist(), et.tolist())}
    assert by_pair[(0, 1)] == RELATION["sector"]
    assert by_pair[(1, 2)] == RELATION["comention"]


def test_overlapping_pair_takes_lowest_relation_id():
    """A pair present in two relations appears once, tagged with the
    lowest-numbered relation (documented tie-break)."""
    tickers = ["AAA", "BBB"]
    ei, et = combine_edges(
        {"correlation": {("AAA", "BBB")}, "sector": {("AAA", "BBB")}}, tickers
    )
    assert ei.shape == (2, 2)                 # one undirected edge, both directions
    assert set(et.tolist()) == {RELATION["sector"]}


def test_combine_edges_drops_tickers_not_in_the_snapshot():
    tickers = ["AAA", "BBB"]
    ei, et = combine_edges({"sector": {("AAA", "ZZZ")}}, tickers)
    assert ei.shape == (2, 0)
    assert et.shape == (0,)


def test_combine_edges_empty_input_is_well_formed():
    ei, et = combine_edges({}, ["AAA", "BBB"])
    assert ei.shape == (2, 0) and ei.dtype == torch.long
    assert et.shape == (0,) and et.dtype == torch.long
