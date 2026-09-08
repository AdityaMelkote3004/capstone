import pandas as pd
import torch

from src.models.finbert_cache import EmbeddingCache, cap_per_day


def _rows(day: str, n: int) -> list:
    return [{"timestamp": pd.Timestamp(f"{day} {h:02d}:00:00"), "headline": f"h{h}"}
            for h in range(n)]


def test_cap_keeps_the_most_recent_n_per_day():
    rows = _rows("2014-06-02", 10)
    capped = cap_per_day(rows, n=3)
    assert len(capped) == 3
    assert {r["headline"] for r in capped} == {"h7", "h8", "h9"}


def test_cap_is_per_day_not_global():
    rows = _rows("2014-06-02", 5) + _rows("2014-06-03", 5)
    capped = cap_per_day(rows, n=2)
    assert len(capped) == 4, "2 per day across 2 days"


def test_cap_leaves_short_days_alone():
    rows = _rows("2014-06-02", 2)
    assert len(cap_per_day(rows, n=10)) == 2


def test_cap_on_empty_input():
    assert cap_per_day([], n=5) == []


def test_cache_roundtrips(tmp_path):
    cache = EmbeddingCache()
    cache.put("a", torch.ones(768))
    cache.put("b", torch.zeros(768))
    path = tmp_path / "emb.pt"
    cache.save(str(path))

    reloaded = EmbeddingCache.load(str(path))
    assert torch.allclose(reloaded.get("a"), torch.ones(768))
    assert reloaded.get("missing") is None


def test_cache_load_of_absent_file_is_empty_not_an_error(tmp_path):
    cache = EmbeddingCache.load(str(tmp_path / "nope.pt"))
    assert cache.get("anything") is None
