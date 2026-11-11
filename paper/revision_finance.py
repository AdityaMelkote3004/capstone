"""Established financial metrics on saved predictions (review round 2).

1. Rank IC: per test day, Spearman correlation between predicted probability and the
   realised target-day return over ALL window-valid stocks (market-neutral by
   construction); mean IC, ICIR = mean / s.d. of daily IC. Per run, then mean over runs.
2. Agreement of the within-day MCC diagnostic with Rank IC across every saved run
   (all tags and folds): Pearson and Spearman correlation over runs.
3. Portfolios (daily rebalanced, equal weight): long-only top-10, long-short
   top-10/bottom-10, market timing (all long if mean prob > 0.5, else all short).
   Costs 0/5/10/20 bps per unit turnover. Annualised return and volatility, Sharpe,
   Sortino, max drawdown, Calmar, mean daily turnover. Equal-weight buy-and-hold as
   reference. Seed-averaged daily return paths are saved for an equity-curve figure.
Writes paper/revision_finance.json.
"""
import glob, json, os, re
import numpy as np
from scipy import stats
from sklearn.metrics import matthews_corrcoef

H = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.path.dirname(H), "results", "experiment11_dtml", "preds")
K, COSTS = 10, (0.0, 0.0005, 0.0010, 0.0020)
NAMES = {"R_mmgt": "MMGT", "R_base": "MMGT-base", "R_profile": "Profile only", "R_lr": "LR",
         "R_mlp": "MLP", "R_lstm": "LSTM", "R_alstm": "ALSTM", "FEDlong": "Federated"}


def runs():
    out = {}
    for f in sorted(glob.glob(os.path.join(P, "*.npz"))):
        m = re.match(r"(.+)_(main|A|B)_s(\d+)\.npz$", os.path.basename(f))
        if m:
            z = np.load(f)
            out.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = {k: z[k] for k in z.files}
    return out


def daily_ic(r):
    ics = []
    for d in np.unique(r["day"]):
        k = r["day"] == d
        if k.sum() > 2:
            ics.append(stats.spearmanr(r["prob"][k], r["ret"][k])[0])
    return np.array(ics)


def within_day(r):
    m = r["label"] >= 0
    d, y, p = r["day"][m], r["label"][m], r["prob"][m]
    v = [matthews_corrcoef(y[d == t], (p[d == t] >= np.median(p[d == t])).astype(int))
         for t in np.unique(d) if len(set(y[d == t])) > 1]
    return float(np.mean(v))


def portfolio(r, kind, cost):
    rets, turns, prev = [], [], np.zeros(87)
    for d in np.unique(r["day"]):
        k = r["day"] == d
        prob, ret, st = r["prob"][k], r["ret"][k] / 100.0, r["stock"][k]
        w = np.zeros(87)
        o = np.argsort(prob)
        if kind == "long_only":
            w[st[o[-K:]]] = 1.0 / K
        elif kind == "long_short":
            w[st[o[-K:]]] = 1.0 / K
            w[st[o[:K]]] = -1.0 / K
        elif kind == "timing":
            w[st] = (1.0 if prob.mean() > 0.5 else -1.0) / len(st)
        else:                                            # buy and hold, equal weight
            w[st] = 1.0 / len(st)
        turn = np.abs(w - prev).sum()
        rets.append(float(np.sum(w[st] * ret)) - cost * turn)
        turns.append(turn)
        prev = w
    return np.array(rets), float(np.mean(turns[1:]))


def risk(rets, turnover):
    curve = np.cumprod(1 + rets)
    dd = float(np.max(1 - curve / np.maximum.accumulate(curve)))
    ann_ret = float((curve[-1]) ** (252 / len(rets)) - 1)
    vol = float(rets.std(ddof=1) * np.sqrt(252))
    down = rets[rets < 0]
    sortino = float(rets.mean() / np.sqrt(np.mean(np.minimum(rets, 0) ** 2)) * np.sqrt(252)) if len(down) else np.inf
    return {"total_return": float(curve[-1] - 1), "ann_return": ann_ret, "ann_vol": vol,
            "sharpe": float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)), "sortino": sortino,
            "max_dd": dd, "calmar": ann_ret / dd if dd > 0 else np.inf, "turnover": turnover}


def main():
    R = runs()
    out = {"ic": {}, "agreement": {}, "portfolio": {}, "paths": {}}
    per_run = []
    for (tag, fold), rr in R.items():
        ics = [daily_ic(r) for r in rr.values()]
        wds = [within_day(r) for r in rr.values()]
        per_run += [(tag, fold, s, float(ic.mean()), w) for s, ic, w in zip(rr, ics, wds)]
        out["ic"][f"{tag}|{fold}"] = {
            "n": len(rr), "ic": [float(np.mean([i.mean() for i in ics])), float(np.std([i.mean() for i in ics], ddof=1)) if len(ics) > 1 else 0.0],
            "icir": float(np.mean([i.mean() / i.std(ddof=1) for i in ics])),
            "ic_pos_days": float(np.mean([(i > 0).mean() for i in ics])),
            "within_day": float(np.mean(wds))}
    ic_all = np.array([p[3] for p in per_run]); wd_all = np.array([p[4] for p in per_run])
    out["agreement"] = {"n_runs": len(per_run), "pearson": float(stats.pearsonr(ic_all, wd_all)[0]),
                        "pearson_p": float(stats.pearsonr(ic_all, wd_all)[1]),
                        "spearman": float(stats.spearmanr(ic_all, wd_all)[0]),
                        "spearman_p": float(stats.spearmanr(ic_all, wd_all)[1]),
                        "runs": per_run}
    for (tag, fold), rr in R.items():
        if fold != "main":
            continue
        name = NAMES.get(tag, tag)
        res = {}
        for kind in ("long_only", "long_short", "timing"):
            for c in COSTS:
                sims = [risk(*portfolio(r, kind, c)) for r in rr.values()]
                res[f"{kind}_{int(c * 1e4)}bps"] = {k: [float(np.mean([s[k] for s in sims])),
                                                        float(np.std([s[k] for s in sims], ddof=1)) if len(sims) > 1 else 0.0]
                                                    for k in sims[0]}
            out["paths"][f"{name}|{kind}"] = np.mean([portfolio(r, kind, 0.0)[0] for r in rr.values()], 0).tolist()
        out["portfolio"][name] = res
    first = next(iter(R[("R_mmgt", "main")].values()))
    bh, bt = portfolio(first, "hold", 0.0)
    out["portfolio"]["Buy and hold"] = {"hold_0bps": {k: [v, 0.0] for k, v in risk(bh, bt).items()}}
    out["paths"]["Buy and hold"] = bh.tolist()
    out["paths"]["dates"] = [str(x) for x in first["dates"][np.unique(first["day"])]]
    json.dump(out, open(os.path.join(H, "revision_finance.json"), "w"), indent=1, default=float)

    print("Rank IC (mean over runs) | ICIR | within-day")
    for k, v in sorted(out["ic"].items(), key=lambda kv: (kv[0].split("|")[1], -kv[1]["ic"][0])):
        print(f"  {k:22} n={v['n']:2} IC={v['ic'][0]:+.4f}±{v['ic'][1]:.4f} ICIR={v['icir']:+.3f} "
              f"pos-days={v['ic_pos_days']:.2f} wd={v['within_day']:+.4f}")
    a = out["agreement"]
    print(f"agreement over {a['n_runs']} runs: pearson {a['pearson']:+.3f} (p={a['pearson_p']:.2g}), "
          f"spearman {a['spearman']:+.3f} (p={a['spearman_p']:.2g})")
    for name, res in out["portfolio"].items():
        for key, v in res.items():
            if key.endswith(("_0bps", "_10bps")):
                print(f"  {name:12} {key:18} Sharpe {v['sharpe'][0]:+.2f} Sortino {v['sortino'][0]:+.2f} "
                      f"annRet {v['ann_return'][0]:+.3f} maxDD {v['max_dd'][0]:.3f} Calmar {v['calmar'][0]:+.2f} turn {v['turnover'][0]:.2f}")


if __name__ == "__main__":
    main()
