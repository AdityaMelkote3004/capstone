"""Review round 5: statistical robustness re-analysis on saved test predictions.

Rules were fixed before any computation (results/experiment11_dtml/PREREGISTRATION.md,
"Review round 5"). No model is retrained, except one post-review reference model
(a logistic regression on the market-index window only), which is cheap and
deterministic and is disclosed as such.

Primary day-level method: hierarchical bootstrap, 2,000 replicates. Each replicate
resamples the training runs of each model with replacement and the 64 test days as a
circular moving-block bootstrap with 5-day blocks. Sensitivity: block lengths 1 and
10, and days-only (runs fixed) with 5-day blocks.

Writes paper/rev5_stats.json.
"""
import glob, json, os, re, sys, warnings

import numpy as np
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
sys.path.insert(0, ROOT)
P = os.path.join(ROOT, "results", "experiment11_dtml", "preds")
B, PRIMARY, BLOCKS = 2000, 5, (1, 5, 10)
warnings.filterwarnings("ignore")

MODELS = {  # name: (tag, seeds or None for all)
    "LR": ("R_lr", None), "MLP": ("R_mlp", None), "LSTM": ("R_lstm", None),
    "ALSTM": ("R_alstm", None), "MMGT-base": ("R_base", None),
    "Profile": ("R_profile", None), "Profile s0-4": ("R_profile", range(5)),
    "MMGT": ("R_mmgt", None), "Federated": ("FEDlong", None),
    "MMGT-D": ("D_final", None), "MMGT-D2": ("D2_final", None),
}


def mcc_from(c):
    tp, tn, fp, fn = c
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def counts(y, pred):
    return np.array([((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 0)).sum(),
                     ((y == 0) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()], float)


def per_day(r, days):
    """Per-day decomposable pieces of every score for one run."""
    D = len(days)
    out = {k: np.zeros((D, 4)) for k in ("cnt", "ccnt", "dcnt")}
    out.update({k: np.full(D, np.nan) for k in ("wd", "ic", "score", "mret", "wd_terc", "wd_all")})
    out["ties"] = 0
    for j, d in enumerate(days):
        k = r["day"] == d
        prob, lab, ret = r["prob"][k].astype(float), r["label"][k], r["ret"][k].astype(float)
        m = lab >= 0
        y, p = lab[m], prob[m]
        out["cnt"][j] = counts(y, (p >= 0.5).astype(int))
        out["ccnt"][j] = counts(y, np.full(len(y), int(p.mean() >= 0.5)))   # stock differences removed
        out["dcnt"][j] = counts(y, (p >= p.mean()).astype(int))              # day level removed
        if len(set(y.tolist())) > 1:
            med = np.median(p)
            out["ties"] += int((p == med).sum() > 1)
            out["wd"][j] = matthews_corrcoef(y, (p >= med).astype(int))
            lo, hi = np.quantile(p, [1 / 3, 2 / 3])
            e = (p <= lo) | (p >= hi)
            if len(set(y[e].tolist())) > 1:
                out["wd_terc"][j] = matthews_corrcoef(y[e], (p[e] >= hi).astype(int))
        f = np.isfinite(ret)
        if f.sum() > 2:
            out["ic"][j] = stats.spearmanr(prob[f], ret[f])[0]
            ya = (ret[f] > 0).astype(int)
            if len(set(ya.tolist())) > 1:
                out["wd_all"][j] = matthews_corrcoef(ya, (prob[f] >= np.median(prob[f])).astype(int))
        out["score"][j] = prob.mean()
        out["mret"][j] = np.nanmean(ret)
    return out


def load(tag, seeds=None, fold="main"):
    runs = []
    for f in sorted(glob.glob(os.path.join(P, f"{tag}_{fold}_s*.npz")),
                    key=lambda s: int(re.search(r"_s(\d+)\.npz$", s).group(1))):
        sd = int(re.search(r"_s(\d+)\.npz$", f).group(1))
        if seeds is not None and sd not in seeds:
            continue
        z = np.load(f)
        r = {k: z[k] for k in ("day", "stock", "prob", "label", "ret")}
        r["day"] = z["dates"][z["day"]]   # align by date: models differ in window length
        runs.append(r)
    return runs


def score(pd_, metric, idx):
    """Score of one run on the (possibly repeated) day indices idx."""
    if metric == "mcc":
        return mcc_from(pd_["cnt"][idx].sum(0))
    if metric == "collapse":
        return mcc_from(pd_["ccnt"][idx].sum(0))
    if metric == "dayremoved":
        return mcc_from(pd_["dcnt"][idx].sum(0))
    if metric in ("wd", "ic", "wd_terc", "wd_all"):
        return float(np.nanmean(pd_[metric][idx]))
    if metric == "daily_auc":
        y = (pd_["mret"][idx] > 0).astype(int)
        return roc_auc_score(y, pd_["score"][idx]) if len(set(y.tolist())) > 1 else 0.5
    if metric == "daily_mcc":
        y = (pd_["mret"][idx] > 0).astype(int)
        return matthews_corrcoef(y, (pd_["score"][idx] >= 0.5).astype(int))
    raise KeyError(metric)


def block_idx(rng, D, L):
    nb = int(np.ceil(D / L))
    starts = rng.integers(0, D, nb)
    return np.concatenate([(s + np.arange(L)) % D for s in starts])[:D]


def boot(pda, pdb, metric, L, hier=True, seed=0):
    """Difference (a - b) of run-averaged scores; pdb None -> level of a."""
    rng = np.random.default_rng(seed)
    D = len(pda[0]["wd"])
    pt = np.mean([score(r, metric, np.arange(D)) for r in pda]) - (
        np.mean([score(r, metric, np.arange(D)) for r in pdb]) if pdb else 0.0)
    draws = []
    for _ in range(B):
        idx = block_idx(rng, D, L)
        ra = [pda[i] for i in rng.integers(0, len(pda), len(pda))] if hier else pda
        v = np.mean([score(r, metric, idx) for r in ra])
        if pdb:
            rb = [pdb[i] for i in rng.integers(0, len(pdb), len(pdb))] if hier else pdb
            v -= np.mean([score(r, metric, idx) for r in rb])
        draws.append(v)
    draws = np.array(draws)
    out = {"point": float(pt), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    if pdb:
        out["p"] = float(max(2 * min((draws <= 0).mean(), (draws >= 0).mean()), 1 / B))
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj.tolist()


def main():
    days = np.unique(load("R_mmgt")[0]["day"])
    PD = {}
    for name, (tag, seeds) in MODELS.items():
        PD[name] = [per_day(r, days) for r in load(tag, seeds)]
    out = {"n_days": int(len(days)), "B": B, "primary_block": PRIMARY}

    # --- sanity: run-level means reproduce the reported values
    full = np.arange(len(days))
    out["levels"] = {n: {m: float(np.mean([score(r, m, full) for r in v]))
                         for m in ("mcc", "wd", "collapse", "ic", "dayremoved", "wd_terc", "wd_all",
                                   "daily_auc", "daily_mcc")} | {"n_runs": len(v)}
                     for n, v in PD.items()}

    # --- comparisons
    comps = [("MMGT", x, "mcc") for x in ("LR", "LSTM", "ALSTM", "MLP", "MMGT-base", "Federated", "MMGT-D", "MMGT-D2")]
    comps += [("MMGT-D2", x, "mcc") for x in ("LR", "LSTM", "ALSTM", "MLP")]
    comps += [("MMGT-D2", "MMGT", m) for m in ("mcc", "wd", "collapse", "ic", "daily_auc")]
    comps += [("MMGT", "MMGT-base", "wd"), ("MMGT", "MMGT-base", "ic"), ("Profile", "MMGT-base", "wd"),
              ("Federated", "Profile s0-4", "wd"), ("Federated", "Profile s0-4", "mcc")]
    res = {}
    for a, b, m in comps:
        key = f"{a} - {b} | {m}"
        res[key] = {f"hier_b{L}": boot(PD[a], PD[b], m, L) for L in BLOCKS}
        res[key]["days_only_b5"] = boot(PD[a], PD[b], m, PRIMARY, hier=False)
        print(key, {k: (round(v["point"], 4), [round(x, 4) for x in v["ci"]], v["p"]) for k, v in res[key].items()})
    fam = [f"MMGT - {x} | mcc" for x in ("LR", "LSTM", "ALSTM", "MLP")]
    adj = holm([res[k][f"hier_b{PRIMARY}"]["p"] for k in fam])
    for k, a_ in zip(fam, adj):
        res[k]["holm_p_primary"] = a_
    out["comparisons"] = res

    # --- levels with primary intervals (supplement)
    out["level_ci"] = {n: {m: boot(PD[n], None, m, PRIMARY)["ci"] for m in ("mcc", "wd", "collapse", "ic", "daily_auc")}
                       for n in ("LR", "MLP", "LSTM", "ALSTM", "MMGT-base", "MMGT", "MMGT-D2")}

    # --- within-day characterisation (label-dependent parts are model-independent)
    any_r = load("R_mmgt")[0]
    scored = sum(1 for d in days if len(set(any_r["label"][(any_r["day"] == d) & (any_r["label"] >= 0)].tolist())) > 1)
    labelled_per_day = [int(((any_r["day"] == d) & (any_r["label"] >= 0)).sum()) for d in days]
    up_share = [float(any_r["label"][(any_r["day"] == d) & (any_r["label"] >= 0)].mean()) for d in days]
    out["within_day"] = {"days_scored": scored, "days_total": int(len(days)),
                         "labelled_per_day": [int(np.min(labelled_per_day)), float(np.median(labelled_per_day)), int(np.max(labelled_per_day))],
                         "days_up_share_above_80_or_below_20": int(sum((u > .8) or (u < .2) for u in up_share)),
                         "ties_days": {n: int(np.mean([r["ties"] for r in v])) for n, v in PD.items()},
                         "per_day_dist": {}}
    for n in ("MMGT-base", "MMGT", "MMGT-D2", "LSTM", "ALSTM"):
        avg = np.nanmean(np.stack([r["wd"] for r in PD[n]]), 0)
        avg = avg[np.isfinite(avg)]
        out["within_day"]["per_day_dist"][n] = {"frac_pos": float((avg > 0).mean()),
                                                "quartiles": [float(x) for x in np.percentile(avg, [25, 50, 75])]}

    # --- agreement correlation with a model-cluster bootstrap (out-of-sample cells)
    runs = json.load(open(os.path.join(H, "revision_finance.json")))["agreement"]["runs"]
    cell = {}
    for t, f, s, ic, wd in runs:
        if f in ("A", "main"):
            cell.setdefault((t, f), []).append((ic, wd))
    keys = sorted(cell)
    models = sorted(set(k[0] for k in keys))
    mean = {k: np.mean(cell[k], 0) for k in keys}
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(B):
        pick = rng.choice(models, len(models))
        pts = np.array([mean[k] for m_ in pick for k in keys if k[0] == m_])
        if len(pts) > 3 and pts[:, 0].std() > 0:
            rs.append(stats.pearsonr(pts[:, 0], pts[:, 1])[0])
    pts = np.array([mean[k] for k in keys])
    out["agreement"] = {"n_cells": len(keys), "n_models": len(models),
                        "pearson": float(stats.pearsonr(pts[:, 0], pts[:, 1])[0]),
                        "spearman": float(stats.spearmanr(pts[:, 0], pts[:, 1])[0]),
                        "model_cluster_ci": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]}
    print("agreement", out["agreement"])

    # --- post-review reference model: logistic regression on the market-index window only
    from sklearn.linear_model import LogisticRegression
    from src.data.acl18 import load_acl18
    d = load_acl18(window=10, fold="main")
    X = d["market"].reshape(len(d["market"]), -1)
    rows = {s: [(i, j) for i in np.where(d["split"] == s)[0] for j in np.where(d["y"][i] >= 0)[0]]
            for s in ("train", "val", "test")}

    def xy(s):
        ii = np.array([i for i, _ in rows[s]]); jj = np.array([j for _, j in rows[s]])
        return X[ii], d["y"][ii, jj], ii
    Xtr, ytr, _ = xy("train"); Xva, yva, _ = xy("val"); Xte, yte, ite = xy("test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    best = None
    for c in (1e-3, 1e-2, 1e-1, 1, 10):
        m_ = LogisticRegression(C=c, max_iter=5000).fit((Xtr - mu) / sd, ytr)
        v = matthews_corrcoef(yva, (m_.predict_proba((Xva - mu) / sd)[:, 1] >= 0.5).astype(int))
        if best is None or v > best[0]:
            best = (v, c, m_)
    pte = best[2].predict_proba((Xte - mu) / sd)[:, 1]
    tdays = np.unique(ite)
    dscore = np.array([pte[ite == i][0] for i in tdays])
    mret = np.array([np.nanmean(d["ret"][i][d["window_ok"][i]]) for i in tdays])
    out["market_only_lr"] = {"C": best[1], "val_mcc": float(best[0]),
                             "test_mcc": float(matthews_corrcoef(yte, (pte >= 0.5).astype(int))),
                             "test_acc": float(((pte >= 0.5).astype(int) == yte).mean()),
                             "daily_auc": float(roc_auc_score((mret > 0).astype(int), dscore)),
                             "pred_up_share": float((pte >= 0.5).mean())}
    print("market-only LR", out["market_only_lr"])

    json.dump(out, open(os.path.join(H, "rev5_stats.json"), "w"), indent=1)
    for n, v in out["levels"].items():
        print(f"{n:13}", {k: round(x, 4) for k, x in v.items()})
    print("within-day", out["within_day"])


if __name__ == "__main__":
    main()
