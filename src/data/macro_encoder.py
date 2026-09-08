"""Daily macro event vectors for Experiment 7.

One vector per trading day, shared by every ticker in that day's cross-section:
FinBERT over market-wide headlines published strictly before the opening bell,
concatenated with FOMC/CPI/NFP release flags.

The pre-open cutoff is the leakage guard, and it is why Task 1 preserves
FNSPID's time-of-day rather than truncating to dates.
"""

import pandas as pd
import torch

from src.models.finbert_cache import cap_per_day

MACRO_DIM = 768


def market_open_utc(day: str) -> pd.Timestamp:
    """09:30 America/New_York on `day`, expressed as a naive UTC timestamp so it
    compares directly against FNSPID's UTC column. Handles EST/EDT."""
    local = pd.Timestamp(f"{day} 09:30:00", tz="America/New_York")
    return local.tz_convert("UTC").tz_localize(None)


def load_macro_calendar(path: str) -> dict:
    """Maps 'YYYY-MM-DD' -> (fomc, cpi, nfp) integer flags.

    FOMC dates are exact (published Federal Reserve schedule). CPI and NFP dates are
    DERIVED BY CONVENTION (BLS mid-month; first Friday) and are NOT verified against the
    BLS release archive -- they may be off by a day or two. Experiment 7's write-up must
    disclose this.
    """
    df = pd.read_csv(path)
    return {
        str(row["date"])[:10]: (int(row["fomc"]), int(row["cpi"]), int(row["nfp"]))
        for _, row in df.iterrows()
    }


def build_macro_vectors(headlines: list, calendar: dict, trading_days: list,
                        cache, lookback_days: int = 3, cap: int = 64) -> dict:
    """Builds {day: (771,) tensor} for every day in `trading_days`.

    A headline counts toward day `t` only if its timestamp is strictly before
    `t`'s market open and within `lookback_days` calendar days of it. Days with
    no qualifying headline get a zero text half rather than being dropped --
    the model owns the learned default, not this cache.
    """
    vectors = {}
    for day in trading_days:
        cutoff = market_open_utc(day)
        window_start = cutoff - pd.Timedelta(days=lookback_days)
        eligible = [h for h in headlines if window_start <= h["timestamp"] < cutoff]
        eligible = cap_per_day(eligible, n=cap)

        if eligible:
            vectors_found = [cache.get(h["headline"]) for h in eligible]
            vectors_found = [v for v in vectors_found if v is not None]
            text = torch.stack(vectors_found).mean(dim=0) if vectors_found else torch.zeros(MACRO_DIM)
        else:
            text = torch.zeros(MACRO_DIM)

        flags = torch.tensor(calendar.get(day, (0, 0, 0)), dtype=torch.float32)
        vectors[day] = torch.cat([text, flags])
    return vectors
