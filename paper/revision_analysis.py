"""Revision analyses on saved test predictions (results/experiment11_dtml/preds).

1. Day-block bootstrap: resample the 64 test DAYS (not stock-days) with
   replacement; for each resample compute every model's seed-averaged MCC on the
   labelled rows of those days; report the 95% interval of MMGT minus each model
   and a two-sided bootstrap p-value. Resampling days respects the correlation of
   stocks on the same day, which the reviewer flagged.
2. Backtests, one per ability the paper separates:
   - long-short (stock selection): each day, equal-weight long the 10 stocks with
     the highest predicted probability and short the 10 lowest, among ALL stocks
     with a valid input window (labelled or not, so no target-day information).
     Dollar-neutral, so market direction cancels.
   - market timing: each day, equal-weight long all stocks if the mean predicted
     probability exceeds 0.5, otherwise short them all.
   Realised close-to-close returns of the target day; costs 0 and 10 bps per unit
   of absolute weight change. Mean +/- s.d. over seeds.
Writes paper/revision_numbers.json.
"""

import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(os.path.dirname(HERE), "results", "experiment11_dtml", "preds")
K, COSTS, B = 10, (0.0, 0.0010), 2000


def load(tag, fold="main"):
    runs = {}
    for f in glob.glob(os.path.join(PRED, f"{tag}_{fold}_s*.npz")):
        seed = int(re.search(r"_s(\d+)\.npz$", f).group(1))
        z = np.load(f)
        runs[seed] = {k: z[k] for k in ["day", "stock", "prob", "label", "ret"]}
    return dict(sorted(runs.items()))


def mcc(r, days=None):
    m = r["label"] >= 0
    if days is not None:
        m &= np.isin(r["day"], days)
    return matthews_corrcoef(r["label"][m], (r["prob"][m] >= 0.5).astype(int))


def bootstrap(runs_a, runs_b, seed=0):
    days = np.unique(next(iter(runs_a.values()))["day"])
    rng = np.random.default_rng(seed)
    # precompute per-run, per-day confusion counts for speed
    def counts(r):
        m = r["label"] >= 0
        d, y, p = r["day"][m], r["label"][m], (r["prob"][m] >= 0.5).astype(int)
        out = {}
        for day in days:
            k = d == day
            yy, pp = y[k], p[k]
            out[day] = np.array([((yy == 1) & (pp == 1)).sum(), ((yy == 0) & (pp == 0)).sum(),
                                 ((yy == 0) & (pp == 1)).sum(), ((yy == 1) & (pp == 0)).sum()])
        return out
    ca = [counts(r) for r in runs_a.values()]
    cb = [counts(r) for r in runs_b.values()]

    def mcc_from(c):
        tp, tn, fp, fn = c
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / den if den > 0 else 0.0
    diffs = []
    for _ in range(B):
        pick = rng.choice(days, len(days))
        a = np.mean([mcc_from(sum(c[d] for d in pick)) for c in ca])
        b = np.mean([mcc_from(sum(c[d] for d in pick)) for c in cb])
        diffs.append(a - b)
    diffs = np.array(diffs)
    point = np.mean([mcc(r) for r in runs_a.values()]) - np.mean([mcc(r) for r in runs_b.values()])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(point), "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)), "p": float(max(p, 1 / B))}


def backtest(r, kind, cost):
    rets, prev_w = [], None
    for day in np.unique(r["day"]):
        k = r["day"] == day
        prob, ret, stock = r["prob"][k], r["ret"][k] / 100.0, r["stock"][k]
        w = np.zeros(87)
        if kind == "long_short":
            order = np.argsort(prob)
            w[stock[order[-K:]]] = 1.0 / K
            w[stock[order[:K]]] = -1.0 / K
        else:
            w[stock] = (1.0 if prob.mean() > 0.5 else -1.0) / len(stock)
        gross = float(np.sum(w[stock] * ret))
        turn = np.abs(w - (prev_w if prev_w is not None else 0)).sum()
        rets.append(gross - cost * turn)
        prev_w = w
    rets = np.array(rets)
    curve = np.cumprod(1 + rets)
    dd = float(np.max(1 - curve / np.maximum.accumulate(curve)))
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {"total_return": float(curve[-1] - 1), "sharpe": sharpe, "max_dd": dd,
            "mean_daily": float(rets.mean())}


def summarise(results):
    keys = results[0].keys()
    return {k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results], ddof=1))
                if len(results) > 1 else 0.0) for k in keys}


def main():
    tags = {"MMGT": "R_mmgt", "MMGT-base": "R_base", "Profile only": "R_profile",
            "MLP": "R_mlp", "LSTM": "R_lstm", "ALSTM": "R_alstm", "LR": "R_lr",
            "Federated": "FEDlong"}
    runs = {name: load(tag) for name, tag in tags.items()}
    runs = {k: v for k, v in runs.items() if v}
    out = {"mcc": {}, "bootstrap_vs_mmgt": {}, "backtest": {}}
    for name, rr in runs.items():
        out["mcc"][name] = [float(mcc(r)) for r in rr.values()]
    if "MMGT" in runs:
        for name, rr in runs.items():
            if name != "MMGT":
                out["bootstrap_vs_mmgt"][name] = bootstrap(runs["MMGT"], rr)
    for name, rr in runs.items():
        out["backtest"][name] = {
            f"{kind}_{int(c * 1e4)}bps": summarise([backtest(r, kind, c) for r in rr.values()])
            for kind in ("long_short", "timing") for c in COSTS}
    json.dump(out, open(os.path.join(HERE, "revision_numbers.json"), "w"), indent=1)
    for name in runs:
        m = out["mcc"][name]
        line = f"{name:12} n={len(m):2} MCC {np.mean(m):+.4f}"
        if name in out["bootstrap_vs_mmgt"]:
            b = out["bootstrap_vs_mmgt"][name]
            line += f" | MMGT-minus: {b['diff']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] p={b['p']:.4f}"
        print(line)
    for name in runs:
        bt = out["backtest"][name]
        print(f"{name:12} long-short Sharpe {bt['long_short_0bps']['sharpe'][0]:+.2f} "
              f"(10bps {bt['long_short_10bps']['sharpe'][0]:+.2f}), "
              f"ret {bt['long_short_0bps']['total_return'][0]:+.3f} | timing Sharpe "
              f"{bt['timing_0bps']['sharpe'][0]:+.2f} (10bps {bt['timing_10bps']['sharpe'][0]:+.2f}), "
              f"ret {bt['timing_0bps']['total_return'][0]:+.3f}")


if __name__ == "__main__":
    main()
