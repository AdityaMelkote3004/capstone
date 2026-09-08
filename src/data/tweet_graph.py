"""Tweet-similarity edges for Experiment 7.

The first graph in this project derived from information that is neither
price-derived nor static: two tickers connect when what people said about them
recently was semantically similar.

Built from the TWEET half of the micro embeddings only, never the news half, so
that graph structure and news content remain separately attributable in the
ablation ladder. Leakage safety comes from Task 4's windowing -- the tweet
vector for day t already excludes day t.
"""

import torch


def build_tweet_edges(micro_embeddings: dict, tickers: list, target_date: str,
                      k: int = 5) -> set:
    """Connects each ticker to its top-`k` most cosine-similar partners by
    tweet embedding, returning undirected alphabetically-sorted pairs.

    Top-k rather than a similarity threshold: Experiment 6 measured a threshold
    at 57% density, which rebuilt the over-smoothing collapse diagnosed in
    Experiment 4. Top-k makes density a controlled parameter.

    Tickers with no tweets that day carry a zero vector and are excluded --
    a zero vector has no meaningful direction to compare.
    """
    present, vectors = [], []
    for ticker in tickers:
        entry = micro_embeddings.get((ticker, target_date))
        if entry is None:
            continue
        tweet_vec = entry[0]
        if float(tweet_vec.abs().sum()) == 0.0:
            continue
        present.append(ticker)
        vectors.append(tweet_vec)

    if len(present) < 2:
        return set()

    matrix = torch.nn.functional.normalize(torch.stack(vectors), dim=1)
    similarity = matrix @ matrix.t()
    similarity.fill_diagonal_(float("-inf"))        # never self-edge

    edges = set()
    neighbours = min(k, len(present) - 1)
    top = similarity.topk(neighbours, dim=1).indices
    for i, ticker in enumerate(present):
        for j in top[i].tolist():
            edges.add(tuple(sorted((ticker, present[j]))))
    return edges
