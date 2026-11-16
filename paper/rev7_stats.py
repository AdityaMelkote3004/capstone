"""Review round 7: market columns, pairwise accuracy, Rank IC by subset, adjusted
returns, alternative market representations, incremental probe, and Experiment 14's
architectural controls.

Rules were fixed before any computation or run (results/experiment11_dtml/
PREREGISTRATION.md, "Experiment 14 / review round 7"). Section (b) uses only saved test
predictions and hidden states; section (a) reads the Experiment 14 control runs once
they exist (it is skipped otherwise).

Populations: S_t = labelled stocks of day t, a subset of V_t = window-valid stocks;
N_t = V_t minus S_t (StockNet's neutral band). M_t = 1[mean_{V_t} r_{i,t} > 0].

Writes paper/rev7_stats.json.
"""
import glob, json, os, re, sys, warnings

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
sys.path.insert(0, ROOT)
sys.path.insert(0, H)
import rev6_stats as R6  # noqa: E402
from rev6_stats import counts, mcc_from, block_idx, EPS  # noqa: E402

B, L = 2000, 5
MIN_SUBSET = 5
warnings.filterwarnings("ignore")

MODELS = dict(R6.MODELS)
CONTROLS = {"D2-unconstrained": "D2_unc", "D2-no-attention": "D2_noatt", "MMGT-wide": "MMGT_wide"}


# ------------------------------------------------------------------ adjusted returns
def adjusted_returns():
    """(date, stock index) -> close-to-close change of the ADJUSTED close (percent),
    feature z_adj_close of the released data; same stock order as load_acl18."""
    root = os.path.join(ROOT, "data", "acl18")
    files = sorted(f for f in os.listdir(os.path.join(root, "ourpped")) if f.endswith(".csv"))
    raw = np.stack([np.genfromtxt(os.path.join(root, "ourpped", f), delimiter=",") for f in files])
    dates = list(np.genfromtxt(os.path.join(root, "trading_dates.csv"), dtype=str))
    miss = (raw[:, :, :11] < -1e5).any(-1)
    adj = np.where(miss, np.nan, raw[:, :, 4])
    return {d: adj[:, j] for j, d in enumerate(dates)}, raw, dates, miss


ADJ, RAW, DATES, MISS = adjusted_returns()


def per_day(r, days):
    out = R6.per_day(r, days)
    D = len(days)
    out["c_pool"] = np.zeros((D, 4))
    for k in ("ic_S", "ic_N", "ic_adj", "mret_adj", "nN"):
        out[k] = np.full(D, np.nan)
    for j, d in enumerate(days):
        k = r["day"] == d
        prob = r["prob"][k].astype(float)
        lab, ret = r["label"][k], r["ret"][k].astype(float)
        S, N = lab >= 0, lab < 0
        out["c_pool"][j] = counts(lab[S], (prob[S] >= 0.5).astype(int))
        out["nN"][j] = N.sum()
        if S.sum() >= MIN_SUBSET:
            out["ic_S"][j] = stats.spearmanr(prob[S], ret[S])[0]
        if N.sum() >= MIN_SUBSET:
            out["ic_N"][j] = stats.spearmanr(prob[N], ret[N])[0]
        ra = ADJ[d][r["stock"][k]]
        f = np.isfinite(ra)
        if f.sum() > 2:
            out["ic_adj"][j] = stats.spearmanr(prob[f], ra[f])[0]
        out["mret_adj"][j] = np.nanmean(ra)
    return out


_score6 = R6.score


def score(pd_, metric, idx):
    if metric == "c_pool":
        return mcc_from(pd_["c_pool"][idx].sum(0))
    if metric in ("ic_S", "ic_N", "ic_adj"):
        return float(np.nanmean(pd_[metric][idx]))
    if metric == "auc_zV_adj":
        y = (pd_["mret_adj"][idx] > 0).astype(int)
        return roc_auc_score(y, pd_["s_zV"][idx]) if len(set(y.tolist())) > 1 else 0.5
    return _score6(pd_, metric, idx)


R6.score = score          # R6.boot calls R6.score; route it through the extended metrics
boot = R6.boot


def welch(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def main():
    days = np.unique(R6.load("R_mmgt")[0]["day"])
    full = np.arange(len(days))
    RUNS = {n: R6.load(t, s) for n, (t, s) in MODELS.items()}
    have_controls = all(len(R6.load(t)) == 10 for t in CONTROLS.values())
    if have_controls:
        RUNS.update({n: R6.load(t) for n, t in CONTROLS.items()})
    PD = {n: [per_day(r, days) for r in v] for n, v in RUNS.items()}
    out = {"n_days": int(len(days)), "B": B, "block": L, "min_subset": MIN_SUBSET,
           "controls_included": have_controls}

    # -------------------------------------------------------------- (b2/b3) checks
    lab_all, adj_all, ret_all = [], [], []
    for r in RUNS["MMGT"][:1]:
        for d in days:
            k = r["day"] == d
            lab_all.append(r["label"][k]); ret_all.append(r["ret"][k])
            adj_all.append(ADJ[d][r["stock"][k]])
    lab_all, adj_all, ret_all = map(np.concatenate, (lab_all, adj_all, ret_all))
    neu = lab_all < 0
    out["neutral_check"] = {
        "n_valid": int(len(lab_all)), "n_neutral": int(neu.sum()),
        "neutral_adj_ret_range": [float(np.nanmin(adj_all[neu])), float(np.nanmax(adj_all[neu]))],
        "neutral_outside_band": int(((adj_all[neu] <= -0.5) | (adj_all[neu] > 0.55)).sum()),
        "labelled_inside_band": int(((adj_all[~neu] > -0.5) & (adj_all[~neu] <= 0.55)).sum()),
        "neutral_per_day": [float(np.mean(PD["MMGT"][0]["nN"])), float(np.min(PD["MMGT"][0]["nN"]))],
        "unadj_vs_adj_abs_diff_test": {"median": float(np.nanmedian(np.abs(ret_all - adj_all))),
                                       "max": float(np.nanmax(np.abs(ret_all - adj_all))),
                                       "n_gt_0.05": int((np.abs(ret_all - adj_all) > 0.05).sum())},
    }
    m0 = PD["MMGT"][0]
    out["neutral_check"]["M_t_days_differ_adj"] = int(((m0["mret"] > 0) != (m0["mret_adj"] > 0)).sum())
    print("neutral/adj check", out["neutral_check"])

    # -------------------------------------------------------------- levels + intervals
    mets = ["c_pool", "wd", "pairacc", "ic", "ic_S", "ic_N", "ic_adj",
            "c_pS", "c_zV", "auc_zV", "auc_pV", "auc_zV_adj"]
    out["levels"], out["level_ci"] = {}, {}
    for n, v in PD.items():
        out["levels"][n] = {m: float(np.mean([score(r, m, full) for r in v])) for m in mets}
        out["levels"][n]["n_runs"] = len(v)
        out["levels"][n]["n_days_ic_N"] = int(np.isfinite(v[0]["ic_N"]).sum())
        out["level_ci"][n] = {m: boot(v, m)["ci"] for m in
                              ("c_pool", "wd", "pairacc", "ic", "ic_S", "ic_N", "c_pS", "c_zV", "auc_zV")}
        print(f"{n:17}", {m: round(x, 3) for m, x in out["levels"][n].items()})

    # -------------------------------------------------------------- paired differences
    pairs = [("MMGT-D2", "MMGT")]
    if have_controls:
        pairs += [(c, "MMGT-D2") for c in CONTROLS] + [(c, "MMGT") for c in CONTROLS]
    out["diffs"] = {}
    for a, b in pairs:
        out["diffs"][f"{a} - {b}"] = {m: boot(PD[a], m, PD[b], m) for m in
                                      ("c_pool", "wd", "pairacc", "ic", "ic_S", "ic_N", "ic_adj",
                                       "c_pS", "c_zV", "auc_zV")}
        print("diff", a, "-", b, {m: (round(x["point"], 3), [round(c, 3) for c in x["ci"]], round(x["p"], 3))
                                  for m, x in out["diffs"][f"{a} - {b}"].items()})

    # -------------------------------------------------------------- run-level (Welch) for controls
    if have_controls:
        def run_metric(n, m):
            return [score(r, m, full) for r in PD[n]]
        out["welch"] = {f"{c} vs {b}": {m: welch(run_metric(c, m), run_metric(b, m)) for m in ("c_pool", "ic", "wd")}
                        for c in CONTROLS for b in ("MMGT-D2", "MMGT")}
        out["run_sd"] = {n: {m: float(np.std(run_metric(n, m), ddof=1)) for m in ("c_pool", "ic")}
                         for n in list(CONTROLS) + ["MMGT", "MMGT-D2"]}
        # day-level identity checks for the unconstrained control
        u = RUNS["D2-unconstrained"]
        out["unconstrained_mean_e_over_V"] = {
            "median_abs": float(np.median(np.concatenate([[abs(r["e"][r["day"] == d].mean()) for d in days] for r in u])))}
        # wording rules R1-R3 (Rank IC, block bootstrap)
        def rule(ctrl, label_contrib, label_not_needed):
            d_main = out["diffs"][f"{ctrl} - MMGT-D2"]["ic"]          # control - D2
            d_mmgt = out["diffs"][f"{ctrl} - MMGT"]["ic"]
            if d_main["ci"][1] < 0:
                return label_contrib
            if d_mmgt["ci"][0] > 0 and d_main["ci"][0] <= 0 <= d_main["ci"][1]:
                return label_not_needed
            return "not resolved"
        out["rules"] = {
            "R1": rule("D2-unconstrained", "the zero-sum constraint contributes",
                       "the gain does not require the constraint"),
            "R2": rule("D2-no-attention", "cross-stock attention contributes",
                       "does not require cross-stock attention"),
            "R3": ("capacity alone does not produce the gain"
                   if out["diffs"]["MMGT-wide - MMGT-D2"]["ic"]["ci"][1] < 0 else "not resolved by this control"),
        }
        print("rules", out["rules"])
        # reproduction check
        rep = glob.glob(os.path.join(ROOT, "results", "experiment11_dtml", "exp14", "D2_final_repro_s0_final.csv"))
        if rep:
            import pandas as pd
            a = pd.read_csv(rep[0]).iloc[0]["test_mcc"]
            b = pd.read_csv(os.path.join(ROOT, "results", "experiment11_dtml", "exp12b_final.csv"))
            b = b[(b.tag == "D2_final") & (b.fold == "main") & (b.seed == 0)].iloc[0]["test_mcc"]
            out["reproduction_D2_final_s0"] = {"rerun": float(a), "original": float(b), "abs_diff": float(abs(a - b))}
            print("repro", out["reproduction_D2_final_s0"])

    # -------------------------------------------------------------- (b5) market representations
    import exp13_probe as E
    from src.data.acl18 import load_acl18
    data = load_acl18(window=10, fold="main")
    T = E.targets(data)
    split = data["split"]
    n_t = len(split)
    feats = RAW[:, :, :11]
    present = ~MISS
    idx_series = np.where(present[..., None], feats, 0.0).sum(0) / np.maximum(present.sum(0), 1)[:, None]
    mret = idx_series[:, 3]                                    # equal-weighted mean close change
    level = np.cumprod(1 + mret / 100.0)
    ind = [mret]
    for w in (5, 10, 15, 20, 25, 30):
        ma = np.array([level[max(0, t - w + 1):t + 1].mean() for t in range(len(level))])
        ind.append((level / ma - 1) * 100)
    ind = np.stack(ind, 1)                                     # days, 7
    tdays = np.arange(10, 10 + n_t)                            # target day t -> raw day index
    win = lambda series: np.stack([series[t - 10:t] for t in tdays]).reshape(n_t, -1)
    reps = {"proxy_11": data["market"].reshape(n_t, -1), "mean_return": win(mret[:, None]), "index_7": win(ind)}
    y = T["next"]
    tr = (split == "train") & np.isfinite(y)
    te = (split == "test") & np.isfinite(y)
    out["market_reps"] = {}
    rng_scores = {}
    for name, X in reps.items():
        p = E.fit_probe(X[tr], y[tr], X[te])
        rng_scores[name] = p
        out["market_reps"][name] = {"daily_auc": float(roc_auc_score(y[te], p)),
                                    "ci_b5": R6.block_auc_ci(y[te], p, 5)}
    print("market reps", out["market_reps"])

    # -------------------------------------------------------------- (b6) incremental probe
    S = E.S
    out["incremental_probe"] = {}
    for tag in ("D2_final", "R_mmgt") + (("D2_noatt",) if have_controls else ()):
        files = sorted(glob.glob(os.path.join(S, f"{tag}_main_s*.npz")))
        if not files:
            continue
        base_s, aug_s, ys = [], [], None
        for f in files:
            z = np.load(f)
            dix, sp = z["day_idx"], z["split"]
            rep = "rep_select_centred" if "rep_select_centred" in z.files else "rep_mmgt"
            Xs = z[rep]
            if rep == "rep_select_centred":
                Xs = Xs[:, Xs.shape[1] // 2:]
            Xm = data["market"][dix].reshape(len(dix), -1)
            yy = y[dix]
            a = (sp == "train") & np.isfinite(yy)
            b = (sp == "test") & np.isfinite(yy)
            base_s.append(E.fit_probe(Xm[a], yy[a], Xm[b]))
            aug_s.append(E.fit_probe(np.hstack([Xm, Xs])[a], yy[a], np.hstack([Xm, Xs])[b]))
            ys = yy[b]
        pb, pa = np.mean(base_s, 0), np.mean(aug_s, 0)
        rng = np.random.default_rng(0)
        dd = []
        for _ in range(B):
            k = block_idx(rng, len(ys), L)
            if len(set(ys[k])) > 1:
                dd.append(roc_auc_score(ys[k], pa[k]) - roc_auc_score(ys[k], pb[k]))
        out["incremental_probe"][tag] = {"rep": rep, "auc_market_only": float(roc_auc_score(ys, pb)),
                                         "auc_market_plus_states": float(roc_auc_score(ys, pa)),
                                         "diff": float(roc_auc_score(ys, pa) - roc_auc_score(ys, pb)),
                                         "diff_ci_b5": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    print("incremental probe", out["incremental_probe"])

    # -------------------------------------------------------------- (a) probe of the no-attention control
    if have_controls:
        out["probe_noatt"] = {}
        files = sorted(glob.glob(os.path.join(S, "D2_noatt_main_s*.npz")))
        for rep in ("select_centred", "select"):
            for tname, t in T.items():
                per_seed, ys = [], None
                for f in files:
                    z = np.load(f)
                    X = z[f"rep_{rep}"]
                    if rep == "select_centred":
                        X = X[:, X.shape[1] // 2:]
                    yy = t[z["day_idx"]]
                    a = (z["split"] == "train") & np.isfinite(yy)
                    b = (z["split"] == "test") & np.isfinite(yy)
                    per_seed.append(E.fit_probe(X[a], yy[a], X[b]))
                    ys = yy[b]
                pm = np.mean(per_seed, 0)
                out["probe_noatt"].setdefault(rep, {})[tname] = {
                    "auc_seedavg": float(roc_auc_score(ys, pm)), "ci_b5": R6.block_auc_ci(ys, pm, 5),
                    "ci_b10": R6.block_auc_ci(ys, pm, 10)}
        print("probe noatt", out["probe_noatt"])

    json.dump(out, open(os.path.join(H, "rev7_stats.json"), "w"), indent=1)
    print("written")


if __name__ == "__main__":
    main()
