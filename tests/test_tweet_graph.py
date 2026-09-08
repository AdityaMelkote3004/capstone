import torch

from src.data.tweet_graph import build_tweet_edges

DAY = "2014-06-03"
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _embeddings(vectors: dict) -> dict:
    """vectors: ticker -> 768-d tweet vector. News half is unused here."""
    return {(t, DAY): (v, torch.zeros(768)) for t, v in vectors.items()}


def _aligned(seed: float) -> torch.Tensor:
    v = torch.zeros(768)
    v[0] = seed
    v[1] = 1.0
    return v


def test_connects_the_similar_tickers():
    emb = _embeddings({
        "AAA": _aligned(1.0), "BBB": _aligned(1.0),      # identical direction
        "CCC": -_aligned(1.0), "DDD": -_aligned(1.0),    # opposite direction
    })
    edges = build_tweet_edges(emb, TICKERS, DAY, k=1)
    assert ("AAA", "BBB") in edges
    assert ("CCC", "DDD") in edges


def test_edges_are_sorted_tuples_without_duplicates():
    emb = _embeddings({t: torch.randn(768) for t in TICKERS})
    edges = build_tweet_edges(emb, TICKERS, DAY, k=2)
    for a, b in edges:
        assert a < b
    assert len(edges) == len(set(edges))


def test_no_self_edges():
    emb = _embeddings({t: torch.randn(768) for t in TICKERS})
    edges = build_tweet_edges(emb, TICKERS, DAY, k=3)
    assert all(a != b for a, b in edges)


def test_top_k_caps_degree():
    emb = _embeddings({t: torch.randn(768) for t in TICKERS})
    k = 2
    edges = build_tweet_edges(emb, TICKERS, DAY, k=k)
    assert len(edges) <= k * len(TICKERS)


def test_tickers_with_no_tweets_contribute_no_edges():
    emb = _embeddings({
        "AAA": _aligned(1.0), "BBB": _aligned(1.0),
        "CCC": torch.zeros(768), "DDD": torch.zeros(768),   # no tweets
    })
    edges = build_tweet_edges(emb, TICKERS, DAY, k=3)
    assert all("CCC" not in e and "DDD" not in e for e in edges)


def test_empty_day_returns_empty_set_not_an_error():
    edges = build_tweet_edges({}, TICKERS, DAY, k=5)
    assert edges == set()


def test_uses_only_the_tweet_half_not_the_news_half():
    """The graph must be attributable to tweets alone - if news leaked into
    the similarity, ablation rows 3 and 4 stop being separable."""
    same_tweets = _aligned(1.0)
    emb = {
        ("AAA", DAY): (same_tweets, torch.randn(768)),
        ("BBB", DAY): (same_tweets, torch.randn(768) * 100),   # wildly different news
        ("CCC", DAY): (-same_tweets, torch.randn(768)),
        ("DDD", DAY): (-same_tweets, torch.randn(768)),
    }
    edges = build_tweet_edges(emb, TICKERS, DAY, k=1)
    assert ("AAA", "BBB") in edges, "identical tweets must still connect"
