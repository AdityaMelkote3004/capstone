import pandas as pd
import torch

from src.data.micro_encoder import build_micro_embeddings, load_ticker_tweets
from src.models.finbert_cache import EmbeddingCache


class FakeCache(EmbeddingCache):
    def get(self, key):
        return torch.full((768,), float(len(str(key)) % 5))


TICKERS = ["AAPL", "MSFT"]
DAYS = ["2014-06-03", "2014-06-04"]


def _news(spec: list) -> list:
    return [{"timestamp": pd.Timestamp(ts), "headline": h, "symbol": s}
            for ts, h, s in spec]


def test_returns_tweet_and_news_vectors_separately():
    tweets = {("AAPL", "2014-06-02"): ["apple tweet"]}
    news = _news([("2014-06-03 10:00:00", "apple headline", "AAPL")])
    out = build_micro_embeddings(tweets, news, TICKERS, DAYS, FakeCache())
    tweet_vec, news_vec = out[("AAPL", "2014-06-03")]
    assert tweet_vec.shape == (768,) and news_vec.shape == (768,)
    assert not torch.allclose(tweet_vec, news_vec), (
        "the two channels must stay distinct - Task 5 builds the graph from "
        "the tweet half alone"
    )


def test_every_ticker_day_pair_is_present():
    out = build_micro_embeddings({}, [], TICKERS, DAYS, FakeCache())
    assert set(out) == {(t, d) for t in TICKERS for d in DAYS}


def test_missing_text_yields_zero_vectors_not_a_crash():
    """The 17 protocol-A samples uncovered by the join must degrade, not fail."""
    out = build_micro_embeddings({}, [], TICKERS, DAYS, FakeCache())
    tweet_vec, news_vec = out[("AAPL", "2014-06-03")]
    assert torch.allclose(tweet_vec, torch.zeros(768))
    assert torch.allclose(news_vec, torch.zeros(768))


def test_leakage_tweets_on_the_target_day_are_excluded():
    same_day = {("AAPL", "2014-06-03"): ["SAME DAY - must be ignored"]}
    a = build_micro_embeddings({}, [], TICKERS, DAYS, FakeCache())
    b = build_micro_embeddings(same_day, [], TICKERS, DAYS, FakeCache())
    assert torch.allclose(a[("AAPL", "2014-06-03")][0],
                          b[("AAPL", "2014-06-03")][0])


def test_leakage_news_after_market_open_is_excluded():
    before = _news([("2014-06-03 10:00:00", "before open", "AAPL")])
    after = before + _news([("2014-06-03 18:00:00", "AFTER OPEN", "AAPL")])
    a = build_micro_embeddings({}, before, TICKERS, DAYS, FakeCache())
    b = build_micro_embeddings({}, after, TICKERS, DAYS, FakeCache())
    assert torch.allclose(a[("AAPL", "2014-06-03")][1],
                          b[("AAPL", "2014-06-03")][1])


def test_news_is_routed_to_the_right_ticker():
    news = _news([("2014-06-03 10:00:00", "apple only report", "AAPL")])
    out = build_micro_embeddings({}, news, TICKERS, DAYS, FakeCache())
    assert not torch.allclose(out[("AAPL", "2014-06-03")][1], torch.zeros(768))
    assert torch.allclose(out[("MSFT", "2014-06-03")][1], torch.zeros(768))


def test_tweet_lookback_window_is_respected():
    old = {("AAPL", "2014-05-01"): ["far too old"]}
    out = build_micro_embeddings(old, [], TICKERS, DAYS, FakeCache(), tweet_lookback=5)
    assert torch.allclose(out[("AAPL", "2014-06-03")][0], torch.zeros(768))


def test_load_ticker_tweets_on_real_data():
    tweets = load_ticker_tweets("dataset/final/stocknet_final_modeling_set.parquet")
    assert len(tweets) > 10000
    key = next(iter(tweets))
    assert isinstance(key, tuple) and len(key) == 2
    assert isinstance(tweets[key], list)


def test_tweet_cap_keeps_the_most_recent_days_not_the_oldest():
    """The window is built most-recent-first (offset=1 is t-1), so the cap must take
    the HEAD. Slicing the tail would keep the stalest tweets in the window, and it
    binds on ~2% of real windows -- the highest-volume, most eventful ticker-days."""
    tweets = {
        ("AAPL", "2014-06-02"): ["recent"] * 3,     # t-1 relative to 2014-06-03
        ("AAPL", "2014-05-30"): ["stale"] * 3,      # older in the same window
    }

    captured = {}

    class RecordingCache(EmbeddingCache):
        def get(self, key):
            captured[key] = captured.get(key, 0) + 1
            return torch.ones(768)

    build_micro_embeddings(tweets, [], ["AAPL"], ["2014-06-03"], RecordingCache(),
                           tweet_lookback=5, cap=3)
    assert "recent" in captured, "the cap must retain the most recent day's tweets"
    assert "stale" not in captured, "the cap must drop the oldest day's tweets first"


def test_leakage_news_exactly_at_market_open_is_excluded():
    """Pins the boundary itself: strictly-before (<), not <=. A headline timestamped
    exactly at the opening bell is realistic for a scheduled release."""
    from src.data.macro_encoder import market_open_utc

    at_open = market_open_utc("2014-06-03")
    before = _news([("2014-06-03 10:00:00", "before open", "AAPL")])
    with_boundary = before + [{"timestamp": at_open, "headline": "AT OPEN", "symbol": "AAPL"}]

    a = build_micro_embeddings({}, before, TICKERS, DAYS, FakeCache())
    b = build_micro_embeddings({}, with_boundary, TICKERS, DAYS, FakeCache())
    assert torch.allclose(a[("AAPL", "2014-06-03")][1], b[("AAPL", "2014-06-03")][1])
