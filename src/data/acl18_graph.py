"""Per-day relation tensors for graph-aware DTML (Experiment 11, stage E).

Combines the project's earlier graph work with DTML's learned stock attention.
Instead of replacing DTML's all-to-all attention with a fixed graph (which the
GAT experiments showed over-smooths), each relation contributes a learned
per-head bias to the attention scores -- or, in "mask" mode, restricts attention
to graph neighbours, which is the old GAT-style design.

Relations, all built from information strictly before the target date:
  sector       static same-sector pairs (Exp 4-6)
  correlation  rolling return-correlation top-k (Exp 6 edge cache)
  comention    tickers co-mentioned in prior news (Exp 6 edge cache)
  residual     the group paper's market-residual graph: each stock's returns over
               the previous 20 trading days are regressed on the equal-weighted
               market return; residuals are correlated and each stock keeps its
               top-8 partners by |corr|, symmetrised.
"""

import numpy as np
import pandas as pd

from src.data.edge_cache import load_edge_cache

FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
EDGE_CACHE = "results/experiment6_dense_graph/edge_cache.pkl"
CACHED_RELATIONS = ("sector", "correlation", "comention")


def residual_graph(tickers, dates, window: int = 20, k: int = 8,
                   root: str = "data/acl18") -> dict:
    # Daily returns come from ACL18's own z_close (column 3), which exists for
    # every trading day. The modeling-set parquet only holds labelled rows, so
    # ~39% of days are absent there and every 20-day window would have gaps.
    import os
    raw = np.stack([np.genfromtxt(os.path.join(root, "ourpped", f"{t}.csv"),
                                  delimiter=",")[:, 3] for t in tickers], axis=1)
    raw[raw < -1e5] = np.nan
    all_days = list(np.genfromtxt(os.path.join(root, "trading_dates.csv"), dtype=str))
    wide = pd.DataFrame(raw, index=all_days, columns=tickers)
    out = {}
    for d in dates:
        # Window is the `window` trading days strictly before d.
        pos = np.searchsorted(all_days, d)
        if pos < window:
            continue
        block = wide.iloc[pos - window:pos].to_numpy()
        good = ~np.isnan(block).any(0)
        if good.sum() < k + 2:
            continue
        r = block[:, good]
        m = r.mean(1, keepdims=True)
        mc = m - m.mean()
        beta = ((r - r.mean(0)) * mc).sum(0) / (mc ** 2).sum()
        resid = r - r.mean(0) - beta * mc
        c = np.corrcoef(resid.T)
        np.fill_diagonal(c, 0.0)
        idx = np.where(good)[0]
        edges = set()
        for a in range(len(idx)):
            for b in np.argsort(-np.abs(c[a]))[:k]:
                i, j = idx[a], idx[b]
                edges.add((min(i, j), max(i, j)))
        out[d] = edges
    return out


def relation_tensor(data: dict, relations) -> np.ndarray:
    """uint8 (days, R, S, S) adjacency aligned to load_acl18()'s grid."""
    tickers, dates = data["tickers"], data["dates"]
    pos = {t: i for i, t in enumerate(tickers)}
    S, D = len(tickers), len(dates)
    adj = np.zeros((D, len(relations), S, S), dtype=np.uint8)
    cache = load_edge_cache(EDGE_CACHE) if any(r in CACHED_RELATIONS for r in relations) else {}
    resid = residual_graph(tickers, dates) if "residual" in relations else {}
    for di, d in enumerate(dates):
        for ri, r in enumerate(relations):
            if r == "residual":
                for i, j in resid.get(d, ()):
                    adj[di, ri, i, j] = adj[di, ri, j, i] = 1
            else:
                for a, b in cache.get(d, {}).get(r, ()):
                    if a in pos and b in pos:
                        i, j = pos[a], pos[b]
                        adj[di, ri, i, j] = adj[di, ri, j, i] = 1
    return adj
