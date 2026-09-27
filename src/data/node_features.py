"""Multi-modal static node features for Experiment 10.

Experiments 4, 5 and 6 put PRICE ONLY on graph nodes. Experiment 7 added text but
never fundamentals. No experiment in this project has given the graph all three
modalities at once, which is what this builds.

Feature groups, and why each is here rather than "because we have it":

  volatility   EWMA (RiskMetrics) and realised vol over 5/10/20 days. This is the
               strongest signal measured anywhere in the project -- EWMA alone
               scores +0.1475 MCC on the volatility target, beating every neural
               model built here. If the graph is to help, it should help on top
               of this, not instead of it.
  fundamentals 13 SEC EDGAR features plus their explicit missing flags. Never
               previously placed on graph nodes.
  tweet counts log1p tweet volume over the window. Across four experiments,
               counts beat FinBERT semantics every time.
  text         PCA-compressed FinBERT tweet embedding. Kept SMALL on purpose:
               Experiment 9 measured that 8 raw-ish dimensions cost a working
               model 37% of its MCC, because dimensions grow while the 20,344
               training rows do not.

LEAKAGE: every feature here is computed from information strictly before the
target date. Volatility uses returns shifted by one day; fundamentals are
forward-filled from filing dates by the upstream pipeline; tweet counts come from
the lookback window columns; PCA is fit on TRAIN rows only by the caller.
"""

import numpy as np
import pandas as pd

FUNDAMENTAL_COLS = [
    "Revenue", "NetIncome", "TotalAssets", "TotalLiabilities",
    "StockholdersEquity", "EPS", "Cash",
    "Revenue_Growth", "NetIncome_Growth", "EPS_Growth",
    "Profit_Margin", "Debt_To_Equity", "ROA",
]
MISSING_COLS = [
    "Revenue_Missing", "NetIncome_Missing", "TotalAssets_Missing",
    "TotalLiabilities_Missing", "StockholdersEquity_Missing",
    "EPS_Missing", "Cash_Missing",
]
COUNT_COLS = [f"Day{d}_TweetCount" for d in range(1, 5)]
EWMA_LAMBDA = 0.94


def build_volatility_features(final_parquet: str) -> pd.DataFrame:
    """Per (Ticker, Date): EWMA and realised vol from STRICTLY PRIOR returns."""
    hist = pd.read_parquet(final_parquet, columns=["Date", "Ticker", "Return"])
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    grouped = hist.groupby("Ticker")["Return"]
    for k in (5, 10, 20):
        # shift(1) is what makes the window end at t-1.
        hist[f"vol_{k}"] = grouped.transform(
            lambda s, k=k: s.shift(1).rolling(k, min_periods=2).std())
    hist["vol_ewma"] = grouped.transform(
        lambda s: np.sqrt(s.shift(1).pow(2).ewm(alpha=1 - EWMA_LAMBDA).mean()))
    hist["ret_lag1"] = grouped.shift(1)
    hist["ret_abs_lag1"] = hist["ret_lag1"].abs()
    return hist.drop(columns=["Return"])


def build_fundamental_features(final_parquet: str) -> pd.DataFrame:
    """Per (Ticker, Date): EDGAR fundamentals and their missing indicators."""
    cols = ["Date", "Ticker"] + FUNDAMENTAL_COLS + MISSING_COLS
    fund = pd.read_parquet(final_parquet, columns=cols)
    fund["Date"] = pd.to_datetime(fund["Date"])
    # Level features span many orders of magnitude across a 87-ticker universe
    # (Revenue in dollars, ROA a ratio). Signed log keeps scale differences from
    # dominating the shared normaliser the way Price3 did in Experiments 3-6.
    for c in ["Revenue", "NetIncome", "TotalAssets", "TotalLiabilities",
              "StockholdersEquity", "Cash"]:
        fund[c] = np.sign(fund[c]) * np.log1p(fund[c].abs())
    return fund


def assemble(exact_df: pd.DataFrame, final_parquet: str,
             text_pca: np.ndarray = None) -> tuple:
    """Returns (matrix, names) of static node features aligned to `exact_df` rows.

    `text_pca` is supplied by the caller already projected and already fit on
    train rows only -- this function never sees the raw embedding, so it cannot
    accidentally fit a transform across splits.
    """
    df = exact_df.copy()
    df["Date"] = pd.to_datetime(df["Target_Date"])

    vol = build_volatility_features(final_parquet)
    fund = build_fundamental_features(final_parquet)
    df = df.merge(vol, on=["Ticker", "Date"], how="left")
    df = df.merge(fund, on=["Ticker", "Date"], how="left")

    blocks, names = [], []

    vol_cols = ["vol_5", "vol_10", "vol_20", "vol_ewma", "ret_lag1", "ret_abs_lag1"]
    blocks.append(df[vol_cols].to_numpy(dtype=np.float32))
    names += vol_cols

    blocks.append(df[FUNDAMENTAL_COLS + MISSING_COLS].to_numpy(dtype=np.float32))
    names += FUNDAMENTAL_COLS + MISSING_COLS

    counts = df[COUNT_COLS].to_numpy(dtype=np.float32).clip(0)
    blocks.append(np.log1p(counts))
    blocks.append(np.log1p(counts.sum(axis=1, keepdims=True)))
    names += [f"log_{c}" for c in COUNT_COLS] + ["log_tweet_total"]

    if text_pca is not None:
        blocks.append(text_pca.astype(np.float32))
        names += [f"text_pc{i}" for i in range(text_pca.shape[1])]

    matrix = np.concatenate(blocks, axis=1)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def standardise(matrix: np.ndarray, is_train: np.ndarray) -> np.ndarray:
    """Z-scores using TRAIN rows only, then clips to +/-10.

    The clip matters: Experiment 3-6 ran with a price feature whose global std was
    inflated 109x by two tickers, flattening everything else to ~0. Fundamentals
    have the same shape of problem across a mixed-cap universe.
    """
    mu = matrix[is_train].mean(axis=0)
    sd = matrix[is_train].std(axis=0) + 1e-8
    return np.clip((matrix - mu) / sd, -10.0, 10.0)


# ---------------------------------------------------------------------------
# Peer features: the graph's information, delivered without message passing.
#
# Experiment 10's screen found message passing contributes ~0 on both targets
# (direction +0.0052, volatility +0.0007) across four conv types. That is a
# statement about the MECHANISM, not necessarily about the information: a GNN
# averages neighbours, and averaging is the wrong operation for a task that
# scores how stocks differ from each other.
#
# These features carry the same neighbourhood information explicitly:
#   - what your peers did yesterday (lead-lag effects between related firms are
#     documented, and a node has never had access to this -- it knows its OWN
#     lagged return, never its neighbours')
#   - how you differ from your peers (the contrast the metric actually rewards)
#   - sector identity, which until now reached the model ONLY through the graph,
#     leaving the no-graph control unfairly deprived
#
# If these help while message passing does not, the graph carries real signal
# and the GNN was simply the wrong way to read it.
# ---------------------------------------------------------------------------

PEER_SOURCE_COLS = ["ret_lag1", "ret_abs_lag1", "vol_ewma"]


def build_peer_features(exact_df: pd.DataFrame, vol_df: pd.DataFrame,
                        edge_cache: dict, sector_csv: str) -> tuple:
    """Per (ticker, day): neighbour means of prior-day quantities, the node's
    contrast against them, neighbour count, and a sector one-hot.

    Leakage: every source column is already shifted to t-1 or earlier, and the
    edges for day t are built from information before t, so neighbour aggregates
    are strictly prior by construction.
    """
    vol = vol_df.copy()
    vol["day"] = pd.to_datetime(vol["Date"]).dt.strftime("%Y-%m-%d")
    by_day = {d: g.set_index("Ticker")[PEER_SOURCE_COLS]
              for d, g in vol.groupby("day")}

    sectors = pd.read_csv(sector_csv)
    sector_of = dict(zip(sectors["Ticker"], sectors["Sector"]))
    sector_names = sorted(set(sector_of.values()))
    sector_index = {s: i for i, s in enumerate(sector_names)}

    days = pd.to_datetime(exact_df["Target_Date"]).dt.strftime("%Y-%m-%d").to_numpy()
    tickers = exact_df["Ticker"].to_numpy()

    n_peer = 2 * len(PEER_SOURCE_COLS) + 1
    out = np.zeros((len(exact_df), n_peer + len(sector_names)), dtype=np.float32)

    neighbour_cache: dict = {}
    for i, (ticker, day) in enumerate(zip(tickers, days)):
        if day not in neighbour_cache:
            adjacency: dict = {}
            for relation_edges in edge_cache.get(day, {}).values():
                for a, b in relation_edges:
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
            neighbour_cache[day] = adjacency
        neighbours = neighbour_cache[day].get(ticker, set())
        frame = by_day.get(day)

        row = np.zeros(n_peer, dtype=np.float32)
        if frame is not None and neighbours:
            present = [t for t in neighbours if t in frame.index]
            if present:
                peer_mean = frame.loc[present].mean(axis=0).to_numpy(dtype=np.float32)
                own = (frame.loc[ticker].to_numpy(dtype=np.float32)
                       if ticker in frame.index else np.zeros(len(PEER_SOURCE_COLS),
                                                              dtype=np.float32))
                row[:len(PEER_SOURCE_COLS)] = peer_mean
                row[len(PEER_SOURCE_COLS):2 * len(PEER_SOURCE_COLS)] = own - peer_mean
                row[-1] = np.log1p(len(present))
        out[i, :n_peer] = row
        out[i, n_peer + sector_index[sector_of[ticker]]] = 1.0

    names = ([f"peer_mean_{c}" for c in PEER_SOURCE_COLS]
             + [f"peer_contrast_{c}" for c in PEER_SOURCE_COLS]
             + ["log_peer_count"]
             + [f"sector_{s}" for s in sector_names])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), names
