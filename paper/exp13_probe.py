"""Experiment 13 -- market-information probe on saved hidden states
(see results/experiment11_dtml/PREREGISTRATION.md, Experiment 13).

Question: does a representation carry information about the market's state?
For each saved run (results/experiment11_dtml/states/{tag}_main_s{seed}.npz) and
each representation, a day-level feature vector [mean_i R, std_i R] over the day's
window-valid stocks is fed to an L2 logistic-regression probe:
  * fitted on TRAINING days only; C chosen by 5-fold time-blocked CV on training
    days (AUC);
  * scored on the 64 TEST days.
For "select_centred" only the std half is used (its mean is zero by construction).

Targets (one per day t):
  next    next-day market direction: mean target-day return over window-valid
          stocks > 0 (what the model is trained to call; weakly predictable)
  yday    yesterday's market direction: mean return of day t-1 over all stocks
          with a return > 0 (fully determined by the inputs)
  trend   10-day market trend: sum over days t-10..t-1 of the mean return > 0
          (fully determined by the inputs)

Reported per (tag, representation, target): mean and s.d. over seeds of the test
AUC; the AUC of the seed-averaged probe score with a 95% day-level bootstrap
interval (2,000 resamples). Pre-registered rule: a representation is "near chance"
for a target if that interval includes 0.5.
Reference row: the raw market-index input window (10 days x 11 features).

Also per tag: Rank IC on test days of the final logit and, for MMGT-D2, of the
timing-branch logit l and the excess logit e.
Writes paper/exp13_probe.json.
"""
import glob, json, os, re, sys, warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
sys.path.insert(0, ROOT)
from src.data.acl18 import load_acl18  # noqa: E402

S = os.path.join(ROOT, "results", "experiment11_dtml", "states")
CS = [1e-3, 1e-2, 1e-1, 1, 10]
B = 2000
warnings.filterwarnings("ignore")


def targets(data):
    ret, ok = data["ret"], data["window_ok"]
    n = len(ret)
    mk_valid = np.array([np.nanmean(ret[i][ok[i]]) if ok[i].any() else np.nan for i in range(n)])
    mk_all = np.array([np.nanmean(ret[i]) if np.isfinite(ret[i]).any() else np.nan for i in range(n)])
    nxt = (mk_valid > 0).astype(float)
    yday = np.full(n, np.nan)
    yday[1:] = (mk_all[:-1] > 0)
    trend = np.full(n, np.nan)
    for i in range(10, n):
        trend[i] = np.nansum(mk_all[i - 10:i]) > 0
    return {"next": nxt, "yday": yday, "trend": trend}


def fit_probe(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-6
    Xtr, Xte = (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]
    best, bestc = -1, CS[0]
    for c in CS:
        aucs = []
        for a, b in KFold(5, shuffle=False).split(Xtr):
            if len(set(ytr[b])) < 2:
                continue
            m = LogisticRegression(C=c, max_iter=2000).fit(Xtr[a], ytr[a])
            aucs.append(roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))
        if np.mean(aucs) > best:
            best, bestc = np.mean(aucs), c
    m = LogisticRegression(C=bestc, max_iter=2000).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def boot_auc(y, p, seed=0):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(B):
        k = rng.integers(0, n, n)
        if len(set(y[k])) > 1:
            vals.append(roc_auc_score(y[k], p[k]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def daily_ic(day, score, ret):
    out = []
    for d in np.unique(day):
        k = (day == d) & np.isfinite(ret)
        if k.sum() > 2:
            out.append(stats.spearmanr(score[k], ret[k])[0])
    return float(np.mean(out))


def main():
    data = load_acl18(window=10, fold="main")
    T = targets(data)
    runs = {}
    for f in sorted(glob.glob(os.path.join(S, "*_main_s*.npz"))):
        m = re.match(r"(.+)_main_s(\d+)\.npz$", os.path.basename(f))
        z = np.load(f)
        runs.setdefault(m.group(1), {})[int(m.group(2))] = {k: z[k] for k in z.files}
    out = {"probe": {}, "rank_ic": {}, "n_test_days": None}

    # reference: raw market-index window
    any_run = next(iter(next(iter(runs.values())).values()))
    days, split = any_run["day_idx"], any_run["split"]
    tr, te = split == "train", split == "test"
    out["n_test_days"] = int(te.sum())
    Xm = data["market"][days].reshape(len(days), -1)
    out["probe"]["market_input"] = {}
    for tname, t in T.items():
        y = t[days]
        a, b = tr & np.isfinite(y), te & np.isfinite(y)
        p = fit_probe(Xm[a], y[a], Xm[b])
        out["probe"]["market_input"][tname] = {"auc_seedavg": float(roc_auc_score(y[b], p)),
                                               "ci": boot_auc(y[b], p), "auc_runs": [None, None]}

    for tag, rr in runs.items():
        reps = sorted(k[4:] for k in next(iter(rr.values())) if k.startswith("rep_"))
        for rep in reps:
            key = f"{tag}/{rep}"
            out["probe"][key] = {}
            for tname, t in T.items():
                per_seed, ys = [], None
                for sd, r in sorted(rr.items()):
                    X = r[f"rep_{rep}"]
                    if rep == "select_centred":
                        X = X[:, X.shape[1] // 2:]
                    y = t[r["day_idx"]]
                    a = (r["split"] == "train") & np.isfinite(y)
                    b = (r["split"] == "test") & np.isfinite(y)
                    per_seed.append(fit_probe(X[a], y[a], X[b]))
                    ys = y[b]
                aucs = [roc_auc_score(ys, p) for p in per_seed]
                pm = np.mean(per_seed, 0)
                out["probe"][key][tname] = {"auc_runs": [float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0],
                                            "auc_seedavg": float(roc_auc_score(ys, pm)),
                                            "ci": boot_auc(ys, pm),
                                            "n_seeds": len(aucs)}
                out["probe"][key][tname]["near_chance"] = bool(out["probe"][key][tname]["ci"][0] <= 0.5 <= out["probe"][key][tname]["ci"][1])
        ic = {}
        for part in ("logit", "l", "e"):
            vals = [daily_ic(r["test_day"], r[f"test_{part}"], r["test_ret"]) for r in rr.values() if f"test_{part}" in r]
            if vals:
                ic[part] = [float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)]
        out["rank_ic"][tag] = ic

    json.dump(out, open(os.path.join(H, "exp13_probe.json"), "w"), indent=1)
    for key, v in out["probe"].items():
        print(f"{key:28}", "  ".join(f"{t}: {v[t]['auc_seedavg']:.3f} [{v[t]['ci'][0]:.2f},{v[t]['ci'][1]:.2f}]"
                                     + (f" runs {v[t]['auc_runs'][0]:.3f}" if v[t]['auc_runs'][0] is not None else "")
                                     for t in T))
    for tag, ic in out["rank_ic"].items():
        print("RankIC", tag, {k: round(x[0], 4) for k, x in ic.items()})


if __name__ == "__main__":
    main()
