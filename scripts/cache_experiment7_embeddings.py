"""Embeds every text Experiment 7 needs, once, into a disk cache.

FinBERT on CPU is the experiment's single largest fixed cost, and the 40-run
ladder must not pay it 40 times. Run this after scripts/fetch_fnspid.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.micro_encoder import load_ticker_tweets
from src.models.finbert_cache import EmbeddingCache, cap_per_day, embed_texts

CACHE_PATH = "data/processed/experiment7_embeddings.pt"
CAP = 64


def main() -> None:
    cache = EmbeddingCache.load(CACHE_PATH)
    print(f"existing cache: {len(cache)} vectors")

    texts = set()
    for name in ("macro", "micro"):
        path = f"data/processed/fnspid_{name}.parquet"
        if not os.path.exists(path):
            print(f"MISSING {path} - run scripts/fetch_fnspid.py first")
            sys.exit(1)
        rows = pd.read_parquet(path).to_dict("records")

        # Apply different capping logic for macro vs micro channels
        if name == "micro":
            capped = []
            by_symbol: dict = {}
            for r in rows:
                by_symbol.setdefault(r["symbol"], []).append(r)
            for symbol_rows in by_symbol.values():
                capped.extend(cap_per_day(symbol_rows, n=CAP))
        else:
            capped = cap_per_day(rows, n=CAP)

        print(f"{name}: {len(rows):,} rows -> {len(capped):,} after per-day cap {CAP}")
        texts.update(r["headline"] for r in capped)

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
