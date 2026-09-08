import pandas as pd
import pytest
import torch

from src.models.finbert_cache import EmbeddingCache, cap_per_day, embed_texts


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


def test_padding_does_not_leak_into_the_pooled_embedding():
    """A short text embedded ALONE must match the same text embedded in a batch
    beside a much longer one. Batching pads the short text; if the attention mask
    were wrong, those pad tokens would be averaged in and the two vectors would
    diverge. This is the property that catches a broken mask."""
    short = "Apple beats earnings."
    long_text = (
        "The Federal Reserve announced a decision today that analysts across the "
        "market had been anticipating for several weeks, citing inflation data "
        "and employment figures."
    )
    try:
        alone = embed_texts([short])
        batched = embed_texts([short, long_text])
    except Exception as exc:                     # no local FinBERT weights
        pytest.skip(f"FinBERT unavailable: {exc}")

    assert alone.shape == (1, 768)
    assert batched.shape == (2, 768)
    diff = (alone[0] - batched[0]).abs().max().item()
    assert diff < 1e-4, (
        f"padding leaked into the mean: max abs diff {diff:.3e}"
    )
