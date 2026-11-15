"""Review round 6: logit/probability alignment, ranking quality and probe intervals.

Rules were fixed before any computation (results/experiment11_dtml/PREREGISTRATION.md,
"Review round 6"). Inputs are the saved test predictions (every window-valid stock;
MMGT-D2 files also hold m_t and e_{i,t}) and the saved hidden states. No model is
retrained. The paper's metric definitions are unchanged; everything here is reported
alongside them.

Populations: S_t = labelled stocks of day t, a subset of V_t = window-valid stocks.
Daily market target: M_t = 1[mean_{i in V_t} r_{i,t} > 0].

Writes paper/rev6_stats.json.
"""
import glob, json, os, re, sys, warnings

import numpy as np
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
sys.path.insert(0, ROOT)
sys.path.insert(0, H)
P = os.path.join(ROOT, "results", "experiment11_dtml", "preds")
B, PRIMARY = 2000, 5
EPS = 1e-6
warnings.filterwarnings("ignore")

MODELS = {  # name: (tag, seeds or None for all)
    "LR": ("R_lr", None), "MLP": ("R_mlp", None), "LSTM": ("R_lstm", None),
    "ALSTM": ("R_alstm", None), "MMGT-base": ("R_base", None),
    "Profile": ("R_profile", None), "MMGT": ("R_mmgt", None),
    "Federated": ("FEDlong", None), "MMGT-D": ("D_final", None), "MMGT-D2": ("D2_final", None),
}


def mcc_from(c):
    tp, tn, fp, fn = c
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def counts(y, pred):
    return np.array([((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 0)).sum(),
                     ((y == 0) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()], float)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def load(tag, seeds=None, fold="main"):
    runs = []
    for f in sorted(glob.glob(os.path.join(P, f"{tag}_{fold}_s*.npz")),
                    key=lambda s: int(re.search(r"_s(\d+)\.npz$", s).group(1))):
        sd = int(re.search(r"_s(\d+)\.npz$", f).group(1))
        if seeds is not None and sd not in seeds:
            continue
        z = np.load(f)
        r = {k: z[k] for k in z.files if k != "dates"}
        r["day"] = z["dates"][z["day"]]          # align by date: models differ in window length
        r["seed"] = sd
        runs.append(r)
    return runs


def per_day(r, days):
    D = len(days)
    out = {k: np.zeros((D, 4)) for k in ("c_pS", "c_pV", "c_zS", "c_zV", "c_m")}
    for k in ("wd", "wdmax", "pairacc", "ic", "s_pV", "s_zV", "s_m", "mret",
              "eV", "eS", "gapS", "gapV", "flipS", "flipV"):
        out[k] = np.full(D, np.nan)
    has_m = "m" in r
    for j, d in enumerate(days):
        k = r["day"] == d
        prob = np.clip(r["prob"][k].astype(float), EPS, 1 - EPS)
        lab, ret = r["label"][k], r["ret"][k].astype(float)
        z = np.log(prob / (1 - prob))
        S = lab >= 0
        y, p = lab[S], prob[S]
        out["c_pS"][j] = counts(y, np.full(len(y), int(p.mean() >= 0.5)))   # defined metric
        out["c_pV"][j] = counts(y, np.full(len(y), int(prob.mean() >= 0.5)))
        out["c_zS"][j] = counts(y, np.full(len(y), int(z[S].mean() >= 0)))
        out["c_zV"][j] = counts(y, np.full(len(y), int(z.mean() >= 0)))
        if len(set(y.tolist())) > 1:
            med = np.median(p)
            out["wd"][j] = matthews_corrcoef(y, (p >= med).astype(int))
            q = y + np.linspace(0.1, 0.4, len(y))          # perfect ranking, distinct values
            out["wdmax"][j] = matthews_corrcoef(y, (q >= np.median(q)).astype(int))
            out["pairacc"][j] = roc_auc_score(y, p)
        f = np.isfinite(ret)
        if f.sum() > 2:
            out["ic"][j] = stats.spearmanr(prob[f], ret[f])[0]
        out["s_pV"][j] = prob.mean()
        out["s_zV"][j] = z.mean()
        out["mret"][j] = np.nanmean(ret)
        if has_m:
            m = float(r["m"][k][0])
            e = r["e"][k].astype(float)
            out["s_m"][j] = m
            out["c_m"][j] = counts(y, np.full(len(y), int(m >= 0)))
            out["eV"][j] = e.mean()
            out["eS"][j] = e[S].mean()
            out["gapS"][j] = abs(p.mean() - sig(m))
            out["gapV"][j] = abs(prob.mean() - sig(m))
            out["flipS"][j] = float((p.mean() >= 0.5) != (m >= 0))
            out["flipV"][j] = float((prob.mean() >= 0.5) != (m >= 0))
    return out


def score(pd_, metric, idx):
    if metric.startswith("c_"):
        return mcc_from(pd_[metric][idx].sum(0))
    if metric in ("wd", "wdmax", "pairacc", "ic"):
        return float(np.nanmean(pd_[metric][idx]))
    if metric == "wd_norm":
        return float(np.nanmean(pd_["wd"][idx]) / np.nanmean(pd_["wdmax"][idx]))
    if metric.startswith("auc_"):
        y = (pd_["mret"][idx] > 0).astype(int)
        s = pd_["s_" + metric[4:]][idx]
        return roc_auc_score(y, s) if len(set(y.tolist())) > 1 else 0.5
    raise KeyError(metric)


def block_idx(rng, D, L):
    nb = int(np.ceil(D / L))
    starts = rng.integers(0, D, nb)
    return np.concatenate([(s + np.arange(L)) % D for s in starts])[:D]


def boot(pda, ma, pdb=None, mb=None, L=PRIMARY, seed=0):
    """Round-5 primary method: runs of each model resampled with replacement, days as
    circular moving blocks; the same day blocks are used for both terms of a
    difference. pdb may be pda itself (two metrics of one model, same runs)."""
    rng = np.random.default_rng(seed)
    D = len(pda[0]["wd"])
    full = np.arange(D)
    same = pdb is pda
    val = lambda runs, m, idx: np.mean([score(r, m, idx) for r in runs])
    pt = val(pda, ma, full) - (val(pdb, mb, full) if pdb is not None else 0.0)
    draws = []
    for _ in range(B):
        idx = block_idx(rng, D, L)
        ra = [pda[i] for i in rng.integers(0, len(pda), len(pda))]
        v = val(ra, ma, idx)
        if pdb is not None:
            rb = ra if same else [pdb[i] for i in rng.integers(0, len(pdb), len(pdb))]
            v -= val(rb, mb, idx)
        draws.append(v)
    draws = np.array(draws)
    out = {"point": float(pt), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    if pdb is not None:
        out["p"] = float(max(2 * min((draws <= 0).mean(), (draws >= 0).mean()), 1 / B))
    return out


def block_auc_ci(y, p, L, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        k = block_idx(rng, len(y), L) if L > 1 else rng.integers(0, len(y), len(y))
        if len(set(y[k])) > 1:
            vals.append(roc_auc_score(y[k], p[k]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    days = np.unique(load("R_mmgt")[0]["day"])
    RUNS = {n: load(t, s) for n, (t, s) in MODELS.items()}
    PD = {n: [per_day(r, days) for r in v] for n, v in RUNS.items()}
    full = np.arange(len(days))
    out = {"n_days": int(len(days)), "B": B, "primary_block": PRIMARY}

    # ---------------------------------------------------------------- levels
    mets = ["c_pS", "c_pV", "c_zS", "c_zV", "auc_pV", "auc_zV", "wd", "wdmax", "wd_norm", "pairacc", "ic"]
    out["levels"] = {}
    for n, v in PD.items():
        lv = {m: float(np.mean([score(r, m, full) for r in v])) for m in mets}
        if n == "MMGT-D2":
            lv.update({m: float(np.mean([score(r, m, full) for r in v])) for m in ("c_m", "auc_m")})
        lv["n_runs"] = len(v)
        # days on which the 0.5 call of the defined aggregation differs from the mean-logit-over-V call
        lv["days_call_differs_pS_vs_zV"] = float(np.mean([
            np.sum([(r["c_pS"][j][0] + r["c_pS"][j][2] > 0) != (r["c_zV"][j][0] + r["c_zV"][j][2] > 0)
                    for j in full]) for r in v]))
        out["levels"][n] = lv
        print(f"{n:10}", {k: round(x, 4) for k, x in lv.items()})

    # ---------------------------------------------------------------- intervals
    out["level_ci"] = {}
    for n in ("LR", "MLP", "LSTM", "ALSTM", "MMGT-base", "Profile", "Federated", "MMGT", "MMGT-D2"):
        ms = ["wd", "c_pS", "c_zV", "auc_pV", "pairacc", "wd_norm"] + (["c_m", "auc_m"] if n == "MMGT-D2" else [])
        out["level_ci"][n] = {m: boot(PD[n], m)["ci"] for m in ms}
        print("ci", n, {m: [round(x, 3) for x in c] for m, c in out["level_ci"][n].items()})

    # ---------------------------------------------------------------- MMGT-D2 alignment
    d2 = PD["MMGT-D2"]
    cat = lambda k: np.concatenate([r[k] for r in d2])
    eV, eS, gS, gV = cat("eV"), cat("eS"), cat("gapS"), cat("gapV")
    q = lambda x: {"median": float(np.nanmedian(x)), "p95": float(np.nanpercentile(x, 95)), "max": float(np.nanmax(x))}
    out["d2_alignment"] = {
        "abs_mean_e_over_V": q(np.abs(eV)),
        "abs_mean_e_over_S": q(np.abs(eS)),
        "gap_meanp_S_vs_sigma_m": q(gS),
        "gap_meanp_V_vs_sigma_m": q(gV),
        "flip_days_per_run_S": [float(np.nansum(r["flipS"])) for r in d2],
        "flip_days_per_run_V": [float(np.nansum(r["flipV"])) for r in d2],
        "collapse_defined_pS": boot(d2, "c_pS"),
        "collapse_market_branch_m": boot(d2, "c_m"),
        "collapse_m_minus_pS": boot(d2, "c_m", d2, "c_pS"),
        "daily_auc_defined_pV": boot(d2, "auc_pV"),
        "daily_auc_market_branch_m": boot(d2, "auc_m"),
        "daily_auc_m_minus_pV": boot(d2, "auc_m", d2, "auc_pV"),
    }
    print("d2 alignment", json.dumps(out["d2_alignment"], indent=None)[:1500])

    # ---------------------------------------------------------------- differences (exploratory)
    out["diffs"] = {
        "D2 - MMGT | pairacc": boot(PD["MMGT-D2"], "pairacc", PD["MMGT"], "pairacc"),
        "D2 - MMGT | wd_norm": boot(PD["MMGT-D2"], "wd_norm", PD["MMGT"], "wd_norm"),
        "D2 - MMGT | c_zV": boot(PD["MMGT-D2"], "c_zV", PD["MMGT"], "c_zV"),
        "MMGT - MMGT-base | pairacc": boot(PD["MMGT"], "pairacc", PD["MMGT-base"], "pairacc"),
        "D2 - ALSTM | pairacc": boot(PD["MMGT-D2"], "pairacc", PD["ALSTM"], "pairacc"),
    }
    print("diffs", out["diffs"])

    # ---------------------------------------------------------------- per-day within-day distribution
    out["wd_per_day"] = {}
    for n, v in PD.items():
        avg = np.nanmean(np.stack([r["wd"] for r in v]), 0)
        avg = avg[np.isfinite(avg)]
        out["wd_per_day"][n] = {"share_pos": float((avg > 0).mean()),
                                "quartiles": [float(x) for x in np.percentile(avg, [25, 50, 75])],
                                "n_days": int(len(avg))}
    mx = PD["MMGT"][0]["wdmax"]
    out["wdmax_per_day"] = {"mean": float(np.nanmean(mx)),
                            "quartiles": [float(x) for x in np.nanpercentile(mx, [25, 50, 75])],
                            "min": float(np.nanmin(mx))}

    # ---------------------------------------------------------------- paired Rank IC by seed
    def run_ic(r):
        v = []
        for d in np.unique(r["day"]):
            k = (r["day"] == d) & np.isfinite(r["ret"])
            if k.sum() > 2:
                v.append(stats.spearmanr(r["prob"][k], r["ret"][k])[0])
        return float(np.mean(v))
    out["paired_ic"] = {}
    rng = np.random.default_rng(0)
    allpairs = []
    for fold in ("main", "A"):
        a = {r["seed"]: run_ic(r) for r in load("D2_final", fold=fold)}
        b = {r["seed"]: run_ic(r) for r in load("R_mmgt", fold=fold)}
        s = sorted(set(a) & set(b))
        dif = np.array([a[x] - b[x] for x in s])
        allpairs += list(dif)
        bs = [rng.choice(dif, len(dif)).mean() for _ in range(10000)]
        out["paired_ic"][fold] = {"seeds": s, "diffs": [float(x) for x in dif], "mean": float(dif.mean()),
                                  "median": float(np.median(dif)), "n_pos": int((dif > 0).sum()),
                                  "wilcoxon_p": float(stats.wilcoxon(dif).pvalue),
                                  "boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    dif = np.array(allpairs)
    bs = [rng.choice(dif, len(dif)).mean() for _ in range(10000)]
    out["paired_ic"]["oos_both"] = {"n": int(len(dif)), "mean": float(dif.mean()), "median": float(np.median(dif)),
                                    "n_pos": int((dif > 0).sum()), "wilcoxon_p": float(stats.wilcoxon(dif).pvalue),
                                    "boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    print("paired ic", out["paired_ic"])

    # ---------------------------------------------------------------- block-length check
    from arch.bootstrap import optimal_block_length
    icd = np.nanmean(np.stack([r["ic"] for r in PD["MMGT-D2"]]), 0) - np.nanmean(np.stack([r["ic"] for r in PD["MMGT"]]), 0)
    acc = []
    for r in RUNS["MMGT"]:
        a_ = []
        for d in days:
            k = (r["day"] == d) & (r["label"] >= 0)
            a_.append(float(((r["prob"][k] >= 0.5).astype(int) == r["label"][k]).mean()))
        acc.append(a_)
    acc = np.mean(acc, 0)
    mret = PD["MMGT"][0]["mret"]
    out["block_length"] = {}
    for name, series in (("d2_minus_mmgt_ic", icd), ("mmgt_daily_accuracy", acc), ("daily_mean_return", mret)):
        s = series[np.isfinite(series)]
        ob = optimal_block_length(s)
        out["block_length"][name] = {"circular": float(ob["circular"].iloc[0]),
                                     "stationary": float(ob["stationary"].iloc[0]),
                                     "acf1": float(np.corrcoef(s[:-1], s[1:])[0, 1])}
    print("block length", out["block_length"])

    # ---------------------------------------------------------------- walk-forward day-weighted averages
    wf = json.load(open(os.path.join(H, "rev4_oos.json")))["walk_forward"]
    nA, nM = 44, 64
    out["wf_day_weighted"] = {n: {k: float((nA * v[k]["A"] + nM * v[k]["main"]) / (nA + nM)) for k in ("mcc", "rank_ic")}
                              | {f"{k}_equal": v[k]["avg_oos"] for k in ("mcc", "rank_ic")}
                              for n, v in wf.items()}

    # ---------------------------------------------------------------- probe intervals with day blocks
    import exp13_probe as E
    from src.data.acl18 import load_acl18
    data = load_acl18(window=10, fold="main")
    T = E.targets(data)
    old = json.load(open(os.path.join(H, "exp13_probe.json")))["probe"]
    sruns = {}
    for f in sorted(glob.glob(os.path.join(E.S, "*_main_s*.npz"))):
        m = re.match(r"(.+)_main_s(\d+)\.npz$", os.path.basename(f))
        z = np.load(f)
        sruns.setdefault(m.group(1), {})[int(m.group(2))] = {k: z[k] for k in z.files}
    any_run = next(iter(next(iter(sruns.values())).values()))
    dix, split = any_run["day_idx"], any_run["split"]
    tr, te = split == "train", split == "test"
    out["probe_block"] = {}

    def record(key, tname, y, p):
        rec = {"auc_seedavg": float(roc_auc_score(y, p)),
               "ci_iid": block_auc_ci(y, p, 1), "ci_b5": block_auc_ci(y, p, 5), "ci_b10": block_auc_ci(y, p, 10)}
        if key in old:
            rec["reproduces_exp13"] = abs(rec["auc_seedavg"] - old[key][tname]["auc_seedavg"]) < 1e-9
        out["probe_block"].setdefault(key, {})[tname] = rec

    Xm = data["market"][dix].reshape(len(dix), -1)
    for tname, t in T.items():
        y = t[dix]
        a, b = tr & np.isfinite(y), te & np.isfinite(y)
        record("market_input", tname, y[b], E.fit_probe(Xm[a], y[a], Xm[b]))
    for tag, rr in sruns.items():
        reps = sorted(k[4:] for k in next(iter(rr.values())) if k.startswith("rep_"))
        for rep in reps:
            for tname, t in T.items():
                per_seed, ys = [], None
                for sd, r in sorted(rr.items()):
                    X = r[f"rep_{rep}"]
                    if rep == "select_centred":
                        X = X[:, X.shape[1] // 2:]
                    y = t[r["day_idx"]]
                    a = (r["split"] == "train") & np.isfinite(y)
                    b = (r["split"] == "test") & np.isfinite(y)
                    per_seed.append(E.fit_probe(X[a], y[a], X[b]))
                    ys = y[b]
                record(f"{tag}/{rep}", tname, ys, np.mean(per_seed, 0))
        print("probe", tag, {k: {t: (round(v["auc_seedavg"], 3), [round(x, 2) for x in v["ci_b5"]])
                                 for t, v in d.items()} for k, d in out["probe_block"].items() if k.startswith(tag)})
    out["probe_block_test_balance"] = {t: float(np.nanmean(T[t][dix][te])) for t in T}

    json.dump(out, open(os.path.join(H, "rev6_stats.json"), "w"), indent=1)
    print("written")


if __name__ == "__main__":
    main()
