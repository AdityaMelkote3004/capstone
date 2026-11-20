"""Scores used in the paper, computed from saved predictions.

Populations on test day t: V_t = every stock with a full input window (all rows of
the demo data), S_t = the labelled ones (label 0 or 1). Neutral-band stocks
(label -1) count only for Rank IC and the market's direction M_t.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def counts(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return np.array([((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 0)).sum(),
                     ((y == 0) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()], float)


def mcc_from(c):
    tp, tn, fp, fn = c
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def auc(y, s):
    """Share of (positive, negative) pairs ordered correctly, ties count half."""
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def day_scores(day):
    """All per-day quantities for one run on one day (a DataFrame of that day's V_t)."""
    p = day["prob"].to_numpy(float)
    y = day["label"].to_numpy()
    r = day["ret"].to_numpy(float)
    S = y >= 0
    yS, pS = y[S], p[S]
    z = logit(p)
    out = {
        "pooled": counts(yS, (pS >= 0.5).astype(int)),
        "collapse": counts(yS, np.full(len(yS), int(pS.mean() >= 0.5))),
        "market_logit": counts(yS, np.full(len(yS), int(z.mean() >= 0))),
        "zbar": z.mean(),
        "mret": np.nanmean(r),
        "wd": np.nan, "pairacc": np.nan,
        "ic": spearman(p[np.isfinite(r)], r[np.isfinite(r)]),
    }
    if len(set(yS.tolist())) > 1:
        out["wd"] = mcc_from(counts(yS, (pS >= np.median(pS)).astype(int)))
        out["pairacc"] = auc(yS, pS)
    return out


def run_scores(run):
    """Paper scores for one training run (all 64 test days)."""
    days = [day_scores(d) for _, d in run.groupby("date", observed=True)]
    M = np.array([d["mret"] > 0 for d in days]).astype(int)
    return {
        "Pooled MCC": mcc_from(sum(d["pooled"] for d in days)),
        "Within-day MCC": float(np.nanmean([d["wd"] for d in days])),
        "Pairwise accuracy": float(np.nanmean([d["pairacc"] for d in days])),
        "Rank IC": float(np.nanmean([d["ic"] for d in days])),
        "Day-collapse MCC": mcc_from(sum(d["collapse"] for d in days)),
        "Market-logit MCC": mcc_from(sum(d["market_logit"] for d in days)),
        "Daily AUC": auc(M, np.array([d["zbar"] for d in days])),
    }


def scoreboard(data):
    """Mean over training runs of every score, per model (the paper's convention)."""
    rows = []
    for model, g in data.groupby("model", observed=True, sort=False):
        runs = [run_scores(r) for _, r in g.groupby("seed", observed=True)]
        row = {"Model": model, "Runs": len(runs)}
        for k in runs[0]:
            vals = [r[k] for r in runs]
            row[k] = float(np.mean(vals))
            if k == "Pooled MCC":
                row["MCC s.d."] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def oracle_mcc(data):
    """Hindsight oracle: every labelled stock gets its day's realised majority label."""
    one = data[data["model"] == data["model"].iloc[0]]
    one = one[one["seed"] == one["seed"].min()]
    c = np.zeros(4)
    for _, d in one.groupby("date", observed=True):
        y = d["label"].to_numpy()
        y = y[y >= 0]
        c += counts(y, np.full(len(y), int(y.mean() >= 0.5)))
    return mcc_from(c)
