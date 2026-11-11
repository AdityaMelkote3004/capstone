"""Extra revision analyses on saved predictions (writes paper/revision_extra.json):
1. Day-block bootstrap of WITHIN-DAY MCC differences (the stock-selection claim).
   Per day: MCC of labels vs (prob >= day's median prob) over labelled stocks,
   exactly as src/utils/metrics.within_day_metrics; days with one class skipped.
   Seed-averaged per-day values are resampled over days.
2. Equal-weight buy-and-hold of all window-valid stocks (reference for the timing
   portfolio) and an always-long timing reference.
3. Within-day gain of MMGT / profile over MMGT-base in each period, with a paired
   test over all runs that share seeds across periods.
"""
import json, os
import numpy as np
from scipy import stats
from sklearn.metrics import matthews_corrcoef
H = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.path.dirname(H), "results", "experiment11_dtml", "preds")
B = 2000

def load(tag, fold, seeds):
    out = {}
    for s in seeds:
        f = os.path.join(P, f"{tag}_{fold}_s{s}.npz")
        if os.path.exists(f):
            z = np.load(f); out[s] = {k: z[k] for k in ["day", "stock", "prob", "label", "ret"]}
    return out

def per_day_wd(r):
    m = r["label"] >= 0
    d, y, p = r["day"][m], r["label"][m], r["prob"][m]
    res = {}
    for day in np.unique(d):
        k = d == day
        if len(set(y[k])) < 2: continue
        res[day] = matthews_corrcoef(y[k], (p[k] >= np.median(p[k])).astype(int))
    return res

def seed_avg_wd(runs):
    per = [per_day_wd(r) for r in runs.values()]
    days = sorted(per[0]); return days, np.array([np.mean([q[d] for q in per]) for d in days])

def boot_wd(ra, rb, seed=0):
    days, a = seed_avg_wd(ra); _, b = seed_avg_wd(rb); diff = a - b
    rng = np.random.default_rng(seed)
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"a": float(a.mean()), "b": float(b.mean()), "diff": float(diff.mean()),
            "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
            "p": float(max(p, 1 / B)), "days": len(days)}

def daily_stats(rets):
    rets = np.array(rets); curve = np.cumprod(1 + rets)
    return {"sharpe": float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)),
            "total_return": float(curve[-1] - 1),
            "max_dd": float(np.max(1 - curve / np.maximum.accumulate(curve)))}

out = {"wd_bootstrap": {}, "reference": {}, "wd_gain_by_period": {}}
main = list(range(10))
mm, base, prof = load("R_mmgt", "main", main), load("R_base", "main", main), load("R_profile", "main", main)
for name, tag, seeds in [("MMGT-base", "R_base", main), ("Profile only", "R_profile", main), ("LSTM", "R_lstm", main),
                         ("ALSTM", "R_alstm", main), ("MLP", "R_mlp", main), ("Federated", "FEDlong", range(5))]:
    out["wd_bootstrap"][f"MMGT-minus-{name}"] = boot_wd(mm, load(tag, "main", seeds))
out["wd_bootstrap"]["Profile-minus-MMGT-base"] = boot_wd(prof, base)
out["wd_bootstrap"]["Federated-minus-Profile-seeds0-4"] = boot_wd(load("FEDlong", "main", range(5)),
                                                                  load("R_profile", "main", range(5)))

r = next(iter(mm.values()))
bh, prev = [], None
for day in np.unique(r["day"]):
    k = r["day"] == day
    bh.append(float(np.mean(r["ret"][k] / 100.0)))
out["reference"]["buy_and_hold_equal_weight"] = daily_stats(bh)

for fold, seeds in [("main", main), ("A", range(5)), ("B", range(5))]:
    m_, b_ = load("R_mmgt", fold, seeds), load("R_base", fold, seeds)
    g = {s: float(np.mean(list(per_day_wd(m_[s]).values())) - np.mean(list(per_day_wd(b_[s]).values())))
         for s in m_ if s in b_}
    out["wd_gain_by_period"][fold] = {"mean_gain": float(np.mean(list(g.values()))), "per_seed": g,
                                      "runs_positive": int(sum(v > 0 for v in g.values())), "n": len(g)}
allg = [v for f in out["wd_gain_by_period"].values() for v in f["per_seed"].values()]
out["wd_gain_by_period"]["all"] = {"mean_gain": float(np.mean(allg)), "n": len(allg),
                                   "runs_positive": int(sum(v > 0 for v in allg)),
                                   "p_one_sample": float(stats.ttest_1samp(allg, 0).pvalue),
                                   "p_sign": float(stats.binomtest(sum(v > 0 for v in allg), len(allg)).pvalue)}
json.dump(out, open(os.path.join(H, "revision_extra.json"), "w"), indent=1, default=str)
for k, v in out["wd_bootstrap"].items():
    print(f"WD {k:28} {v['a']:+.4f} vs {v['b']:+.4f} diff {v['diff']:+.4f} [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p']:.3f} days={v['days']}")
print("buy&hold", out["reference"])
for f, v in out["wd_gain_by_period"].items():
    print(f, {k: v[k] for k in v if k != "per_seed"})
