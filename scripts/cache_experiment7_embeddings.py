"""Embeds every text Experiment 7 needs, once, into a disk cache.

FinBERT on CPU is the experiment's single largest fixed cost, and the 40-run
ladder must not pay it 40 times. Run this after scripts/fetch_fnspid.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.macro_encoder import market_open_utc
from src.data.micro_encoder import load_ticker_tweets
from src.models.finbert_cache import EmbeddingCache, cap_per_day, embed_texts

CACHE_PATH = "data/processed/experiment7_embeddings.pt"
CAP = 64
MACRO_LOOKBACK_DAYS = 3


def macro_texts_needed(macro_rows: list, trading_days: list, cap: int) -> set:
    """The exact set of macro headlines build_macro_vectors will ask for.

    Derived by replicating the encoder's windowing rather than approximating it with a
    per-calendar-day cap. Those two selections diverge on the target day: a plain per-day
    cap keeps that day's LAST headlines, which are published after the opening bell, while
    the encoder wants the last ones published strictly BEFORE it. Approximating here left
    24% of requested macro embeddings uncached, and a cache miss degrades silently to a
    zero vector rather than raising.
    """
    needed = set()
    for day in trading_days:
        cutoff = market_open_utc(day)
        start = cutoff - pd.Timedelta(days=MACRO_LOOKBACK_DAYS)
        eligible = [h for h in macro_rows if start <= h["timestamp"] < cutoff]
        for h in cap_per_day(eligible, n=cap):
            needed.add(h["headline"])
    return needed


def main() -> None:
    cache = EmbeddingCache.load(CACHE_PATH)
    print(f"existing cache: {len(cache)} vectors")

    texts = set()

    # Process macro channel with exact encoder windowing
    macro_path = "data/processed/fnspid_macro.parquet"
    if not os.path.exists(macro_path):
        print(f"MISSING {macro_path} - run scripts/fetch_fnspid.py first")
        sys.exit(1)
    macro_rows = pd.read_parquet(macro_path).to_dict("records")

    # Load trading days from protocol file
    protocol_path = "dataset/stocknet_protocol_a_exact.parquet"
    if not os.path.exists(protocol_path):
        print(f"MISSING {protocol_path}")
        sys.exit(1)
    exact = pd.read_parquet(protocol_path, columns=["Target_Date"])
    trading_days = sorted(pd.to_datetime(exact["Target_Date"]).dt.strftime("%Y-%m-%d").unique())

    # Get macro texts using exact encoder windowing
    macro_headlines = macro_texts_needed(macro_rows, trading_days, CAP)
    print(f"macro: {len(macro_rows):,} rows -> {len(macro_headlines):,} unique headlines (exact encoder windowing)")
    texts.update(macro_headlines)

    # Process micro channel (unchanged)
    micro_path = "data/processed/fnspid_micro.parquet"
    if not os.path.exists(micro_path):
        print(f"MISSING {micro_path} - run scripts/fetch_fnspid.py first")
        sys.exit(1)
    micro_rows = pd.read_parquet(micro_path).to_dict("records")
    capped_micro = []
    by_symbol: dict = {}
    for r in micro_rows:
        by_symbol.setdefault(r["symbol"], []).append(r)
    for symbol_rows in by_symbol.values():
        capped_micro.extend(cap_per_day(symbol_rows, n=CAP))
    print(f"micro: {len(micro_rows):,} rows -> {len(capped_micro):,} after per-symbol per-day cap {CAP}")
    texts.update(r["headline"] for r in capped_micro)

    tweets = load_ticker_tweets("dataset/final/stocknet_final_modeling_set.parquet")
    for segments in tweets.values():
        texts.update(segments[-CAP:])
    print(f"unique texts to embed: {len(texts):,}")

    pending = [t for t in sorted(texts) if cache.get(t) is None]
    print(f"already cached: {len(texts) - len(pending):,} | to embed: {len(pending):,}")

    batch = 256
    total_batches = (len(pending) + batch - 1) // batch
    for index, start in enumerate(range(0, len(pending), batch)):
        chunk = pending[start:start + batch]
        for text, vector in zip(chunk, embed_texts(chunk)):
            cache.put(text, vector)
        if (index + 1) % 10 == 0 or (index + 1) == total_batches:
            cache.save(CACHE_PATH)      # atomic; every 10th batch and at the end
        print(f"  {min(start + batch, len(pending)):,}/{len(pending):,}", flush=True)

    print(f"done. cache holds {len(cache)} vectors at {CACHE_PATH}")


if __name__ == "__main__":
    main()
