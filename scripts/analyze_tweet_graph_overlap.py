"""Spec Component 7, item 3: is the learned tweet graph structurally NEW?

Experiment 7's row 3 beats the sector graph (row 8) on within-day MCC. That is a
performance claim; this script asks the structural question behind it -- does the
tweet graph connect pairs the hand-designed graphs already connect, or different
ones? If its edges were mostly a subset of sector cliques, "learned from tweets"
would be a re-derivation of sector membership by a slower route.

Jaccard against the sector graph (static) and the 60-day return-correlation graph
(rebuilt per day), averaged over test days.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.graph_edges import build_correlation_edges, build_sector_edges, load_returns_matrix
from src.data.micro_encoder import build_micro_embeddings, load_ticker_tweets
from src.data.tweet_graph import build_tweet_edges
from src.models.finbert_cache import EmbeddingCache

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
CACHE_PATH = "data/processed/experiment7_embeddings.pt"
OUT = "results/experiment7_macro_conditioned/tweet_graph_overlap.json"
TWEET_LOOKBACK = 20


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main() -> None:
    df = pd.read_parquet(EXACT)
    tickers = sorted(df["Ticker"].unique())
    days = sorted(pd.to_datetime(df["Target_Date"]).dt.strftime("%Y-%m-%d").unique())
    test_days = sorted(pd.to_datetime(df[df.Split == "test"]["Target_Date"])
                       .dt.strftime("%Y-%m-%d").unique())

    print("building micro embeddings (needed for tweet edges)...")
    cache = EmbeddingCache.load(CACHE_PATH)
    micro_rows = pd.read_parquet("data/processed/fnspid_micro.parquet").to_dict("records")
    tweets = load_ticker_tweets(FINAL)
    micro = build_micro_embeddings(tweets, micro_rows, tickers, days, cache,
                                   tweet_lookback=TWEET_LOOKBACK)

    sector = build_sector_edges(SECTOR_CSV)
    returns = load_returns_matrix(FINAL)

    stats = []
    for day in test_days:
        tweet = build_tweet_edges(micro, tickers, day)
        corr = build_correlation_edges(returns, pd.Timestamp(day))
        if not tweet:
            continue
        stats.append({
            "day": day,
            "tweet_edges": len(tweet),
            "sector_edges": len(sector),
            "corr_edges": len(corr),
            "jaccard_sector": jaccard(tweet, sector),
            "jaccard_corr": jaccard(tweet, corr),
            "frac_tweet_in_sector": len(tweet & sector) / len(tweet),
            "frac_tweet_in_corr": len(tweet & corr) / len(tweet),
        })

    s = pd.DataFrame(stats)

    # A Jaccard is only interpretable against what two UNRELATED graphs of these
    # densities would score. For random edge sets over C(87,2)=3741 pairs,
    # E|A n B| = |A||B|/3741, so E[J] = that over |A|+|B|-E|A n B|.
    pairs = len(tickers) * (len(tickers) - 1) / 2
    def expected_jaccard(a, b):
        inter = a * b / pairs
        return inter / (a + b - inter)

    summary = {
        "test_days_scored": len(s),
        "mean_tweet_edges": round(s.tweet_edges.mean(), 1),
        "sector_edges": int(s.sector_edges.iloc[0]),
        "mean_corr_edges": round(s.corr_edges.mean(), 1),
        "mean_jaccard_vs_sector": round(s.jaccard_sector.mean(), 4),
        "mean_jaccard_vs_correlation": round(s.jaccard_corr.mean(), 4),
        "mean_frac_tweet_edges_also_sector": round(s.frac_tweet_in_sector.mean(), 4),
        "mean_frac_tweet_edges_also_corr": round(s.frac_tweet_in_corr.mean(), 4),
        "possible_pairs": int(pairs),
        "random_baseline_jaccard_vs_sector":
            round(expected_jaccard(s.tweet_edges.mean(), s.sector_edges.mean()), 4),
        "random_baseline_jaccard_vs_correlation":
            round(expected_jaccard(s.tweet_edges.mean(), s.corr_edges.mean()), 4),
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    pd.Series(summary).to_json(OUT, indent=2)
    s.to_csv(OUT.replace(".json", "_by_day.csv"), index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
