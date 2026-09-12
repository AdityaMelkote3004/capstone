"""Per-(ticker, day) micro signal for Experiment 7.

Two channels, deliberately kept separate:

  tweet_vec -- FinBERT over the ticker's tweets in a window ending at t-1
  news_vec  -- FinBERT over the ticker's FNSPID headlines before t's open

They are returned as a pair rather than pooled because Task 5 builds the
tweet-similarity graph from the tweet half ALONE. Pooling here would make graph
structure and news content inseparable, and the ablation ladder would silently
stop isolating what it claims to isolate.
"""

import collections

import pandas as pd
import torch

from src.data.macro_encoder import market_open_utc
from src.models.finbert_cache import cap_per_day

MICRO_DIM = 768


def load_ticker_tweets(final_parquet: str) -> dict:
    """Maps (ticker, 'YYYY-MM-DD') -> list of tweet segment strings, pooling
    Company_Texts and Event_Texts. Rows with no text are skipped."""
    df = pd.read_parquet(final_parquet,
                         columns=["Date", "Ticker", "Company_Texts", "Event_Texts"])
    df["Date"] = pd.to_datetime(df["Date"])
    out: dict = collections.defaultdict(list)
    for date, ticker, company, event in zip(df["Date"], df["Ticker"],
                                            df["Company_Texts"], df["Event_Texts"]):
        key = (ticker, date.strftime("%Y-%m-%d"))
        for blob in (company, event):
            if blob is None or len(str(blob)) <= 2:
                continue
            out[key].extend(s for s in str(blob).split("[SEP]") if len(s.strip()) > 2)
    return dict(out)


def _mean_or_zero(texts: list, cache) -> torch.Tensor:
    if not texts:
        return torch.zeros(MICRO_DIM)
    vectors = [cache.get(t) for t in texts]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return torch.zeros(MICRO_DIM)
    return torch.stack(vectors).mean(dim=0)


def build_micro_embeddings(tweets_by_key: dict, news_rows: list, tickers: list,
                           trading_days: list, cache, tweet_lookback: int = 20,
                           cap: int = 64) -> dict:
    """Builds {(ticker, day): (tweet_vec, news_vec)} for every pair.

    Tweets use a window ending at t-1 (strictly excluding the target day);
    news uses everything published before t's market open. Missing text yields
    zero vectors -- the 17 protocol-A samples uncovered by the tweet join must
    degrade to an empty neighbourhood, never crash and never be dropped.
    """
    news_by_symbol: dict = collections.defaultdict(list)
    for row in news_rows:
        news_by_symbol[row["symbol"]].append(row)

    out = {}
    for day in trading_days:
        day_ts = pd.Timestamp(day)
        cutoff = market_open_utc(day)
        window_start = day_ts - pd.Timedelta(days=tweet_lookback)

        for ticker in tickers:
            texts = []
            for offset in range(1, tweet_lookback + 1):
                past = (day_ts - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if pd.Timestamp(past) >= window_start:
                    texts.extend(tweets_by_key.get((ticker, past), []))

            eligible_news = [n for n in news_by_symbol.get(ticker, [])
                             if n["timestamp"] < cutoff]
            eligible_news = cap_per_day(eligible_news, n=cap)

            out[(ticker, day)] = (
                _mean_or_zero(texts[:cap], cache),
                _mean_or_zero([n["headline"] for n in eligible_news], cache),
            )
    return out
