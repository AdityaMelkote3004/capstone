import pandas as pd
import torch

from src.data.macro_encoder import (build_macro_vectors, load_macro_calendar,
                                    market_open_utc)
from src.models.finbert_cache import EmbeddingCache


class FakeCache(EmbeddingCache):
    """Deterministic stand-in so tests never invoke FinBERT."""

    def get(self, key):
        return torch.full((768,), float(len(key) % 7))


def _headlines(spec: list) -> list:
    return [{"timestamp": pd.Timestamp(ts), "headline": h} for ts, h in spec]


CALENDAR = {"2014-06-02": (1, 0, 0)}
DAYS = ["2014-06-02", "2014-06-03"]


def test_market_open_is_0930_eastern_in_utc():
    # 2014-06-02 is EDT (UTC-4), so 09:30 ET == 13:30 UTC
    assert market_open_utc("2014-06-02") == pd.Timestamp("2014-06-02 13:30:00")


def test_market_open_handles_winter_standard_time():
    # 2014-01-02 is EST (UTC-5), so 09:30 ET == 14:30 UTC
    assert market_open_utc("2014-01-02") == pd.Timestamp("2014-01-02 14:30:00")


def test_load_macro_calendar(tmp_path):
    path = tmp_path / "cal.csv"
    path.write_text("date,fomc,cpi,nfp\n2014-06-02,1,0,1\n")
    cal = load_macro_calendar(str(path))
    assert cal["2014-06-02"] == (1, 0, 1)


def test_vector_shape_and_calendar_flags_are_appended():
    heads = _headlines([("2014-06-02 10:00:00", "pre-open news")])
    vectors = build_macro_vectors(heads, CALENDAR, DAYS, FakeCache())
    assert vectors["2014-06-02"].shape == (771,)
    assert vectors["2014-06-02"][-3:].tolist() == [1.0, 0.0, 0.0]


def test_leakage_headlines_at_or_after_market_open_are_excluded():
    """THE leakage test: news published after the bell cannot inform that day."""
    clean = _headlines([("2014-06-02 10:00:00", "before open")])
    poisoned = clean + _headlines([
        ("2014-06-02 14:00:00", "AFTER OPEN - must be ignored"),
        ("2014-06-02 20:00:00", "ALSO AFTER - must be ignored"),
    ])
    a = build_macro_vectors(clean, CALENDAR, DAYS, FakeCache())
    b = build_macro_vectors(poisoned, CALENDAR, DAYS, FakeCache())
    assert torch.allclose(a["2014-06-02"], b["2014-06-02"])


def test_lookback_window_is_respected():
    old = _headlines([("2014-05-01 10:00:00", "far too old")])
    vectors = build_macro_vectors(old, CALENDAR, DAYS, FakeCache(), lookback_days=3)
    # no qualifying headline -> text half is the zero default
    assert torch.allclose(vectors["2014-06-02"][:768], torch.zeros(768))


def test_days_with_no_headlines_still_produce_a_vector():
    vectors = build_macro_vectors([], CALENDAR, DAYS, FakeCache())
    assert set(vectors) == set(DAYS)
    for day in DAYS:
        assert vectors[day].shape == (771,)
        assert torch.isfinite(vectors[day]).all()


def test_calendar_flags_default_to_zero_for_unlisted_days():
    vectors = build_macro_vectors([], CALENDAR, DAYS, FakeCache())
    assert vectors["2014-06-03"][-3:].tolist() == [0.0, 0.0, 0.0]


def test_cache_misses_degrade_to_the_zero_default_not_a_crash():
    """The real EmbeddingCache returns None for an unseen headline; a miss must not
    raise, because the per-day cap means some headlines are deliberately unembedded."""
    class MissingCache(EmbeddingCache):
        def get(self, key):
            return None

    heads = _headlines([("2014-06-02 10:00:00", "never embedded")])
    vectors = build_macro_vectors(heads, CALENDAR, DAYS, MissingCache())
    assert torch.allclose(vectors["2014-06-02"][:768], torch.zeros(768))
    assert vectors["2014-06-02"][-3:].tolist() == [1.0, 0.0, 0.0]
