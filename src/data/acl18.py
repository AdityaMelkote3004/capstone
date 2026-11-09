"""ACL18 as preprocessed by Adv-ALSTM (Feng et al., IJCAI 2019) -- the exact data
DTML (Yoo et al., KDD 2021) reports on.

Source: github.com/fulifeng/Adv-ALSTM, data/stocknet-dataset/price/ourpped.
87 per-ticker CSVs on a shared 652-day calendar (2013-06-03 to 2015-12-31).
Columns: 11 features, then the label, then the raw adjusted close.

  features  z_open, z_high, z_low, z_close, z_adj_close, z_d5 .. z_d30
            (DTML Table 3; percent changes and moving-average ratios)
  label     +1 / -1, 0 for StockNet's buffer zone, -123321 for missing rows

Alignment was checked before any model was written: the SAME row's z_close
correlates +0.81 with that row's label (the label is that day's move), while the
PREVIOUS row's z_close correlates -0.006. Windows here therefore end at t-1 --
the same rule Adv-ALSTM's own loader applies (rows [t-w, t)).

Splits are Adv-ALSTM's: train from 2014-01-02, val from 2015-08-03, test from
2015-10-01. Test is 64 days / 3,720 labelled samples, identical in count and
dates to this project's exact-protocol test set.
"""

import os

import numpy as np

MISSING = -1e5
N_FEATURES = 11
SPLIT_STARTS = {"train": "2014-01-02", "val": "2015-08-03", "test": "2015-10-01"}


def load_acl18(root: str = "data/acl18", window: int = 15) -> dict:
    files = sorted(f for f in os.listdir(os.path.join(root, "ourpped"))
                   if f.endswith(".csv"))
    raw = np.stack([np.genfromtxt(os.path.join(root, "ourpped", f), delimiter=",")
                    for f in files])                      # stocks, days, 13
    dates = list(np.genfromtxt(os.path.join(root, "trading_dates.csv"), dtype=str))

    feats = raw[:, :, :N_FEATURES]
    row_missing = (feats < MISSING).any(-1)               # stocks, days
    label = raw[:, :, -2]
    labelled = (np.abs(np.abs(label) - 1.0) < 1e-6)       # exactly +1 or -1

    n_stocks, n_days = row_missing.shape
    target_days = np.arange(window, n_days)
    x = np.zeros((len(target_days), n_stocks, window, N_FEATURES), dtype=np.float32)
    window_ok = np.zeros((len(target_days), n_stocks), dtype=bool)
    y = np.full((len(target_days), n_stocks), -1, dtype=np.int64)

    for i, t in enumerate(target_days):
        ok = ~row_missing[:, t - window:t].any(axis=1)
        window_ok[i] = ok
        x[i, ok] = feats[ok, t - window:t]
        has = ok & labelled[:, t] & ~row_missing[:, t]
        y[i, has] = (label[has, t] > 0).astype(np.int64)

    # DTML builds its global context from a market index (S&P 500 for ACL18).
    # That series is not in the Adv-ALSTM data, so the index is proxied by the
    # equal-weighted cross-sectional mean of the same 11 features over the stocks
    # present each day -- the same market factor the friend's paper residualises
    # against. Every input to it ends at t-1, so it carries no label information.
    present = ~row_missing
    safe = np.where(present[..., None], feats, 0.0)
    index_series = safe.sum(0) / np.maximum(present.sum(0), 1)[:, None]
    market = np.stack([index_series[t - window:t] for t in target_days]).astype(np.float32)

    day_dates = [dates[t] for t in target_days]
    split = np.empty(len(target_days), dtype=object)
    for i, d in enumerate(day_dates):
        if d >= SPLIT_STARTS["test"]:
            split[i] = "test"
        elif d >= SPLIT_STARTS["val"]:
            split[i] = "val"
        elif d >= SPLIT_STARTS["train"]:
            split[i] = "train"
        else:
            split[i] = "warmup"

    return {"x": x, "window_ok": window_ok, "y": y, "market": market,
            "dates": day_dates, "split": split, "tickers": [f[:-4] for f in files]}
