"""Review round 4: quarantine walk-forward period B.

Period B (Aug-Sep 2015) is the validation period of the standard split, on which the
settings reused in walk-forward were chosen, so it is not out-of-sample. This script
recomputes every pooled walk-forward statistic in the paper over the two
out-of-sample periods only (A: Jun-Jul 2015; standard: Oct-Dec 2015), and reports the
B values separately. No new training; inputs are the saved results.

Writes paper/rev4_oos.json.
"""
import collections, json, os

import numpy as np
import pandas as pd
from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(H), "results", "experiment11_dtml")
OOS = ("A", "main")


def main():
    tabs = json.load(open(os.path.join(H, "revision_tables.json")))
    runs = json.load(open(os.path.join(H, "revision_finance.json")))["agreement"]["runs"]
    extra = json.load(open(os.path.join(H, "revision_extra.json")))
    out = {}

    # --- walk-forward table: MCC and Rank IC per period, average over A + standard
    mcc = {}
    for name in ("LR", "MLP", "LSTM", "MMGT-base", "MMGT"):
        for f in ("A", "B", "main"):
            mcc[(name, f)] = tabs[f][name]["test_mcc"][0]
    for csv, tag, name in (("exp12_final.csv", "D_final", "MMGT-D"), ("exp12b_final.csv", "D2_final", "MMGT-D2")):
        d = pd.read_csv(os.path.join(R, csv))
        for f in ("A", "B", "main"):
            mcc[(name, f)] = float(d[(d.tag == tag) & (d.fold == f)].test_mcc.mean())
    tagname = {"R_lr": "LR", "R_mlp": "MLP", "R_lstm": "LSTM", "R_base": "MMGT-base",
               "R_mmgt": "MMGT", "D_final": "MMGT-D", "D2_final": "MMGT-D2"}
    cell = collections.defaultdict(list)
    for t, f, s, ic, wd in runs:
        cell[(t, f)].append((s, ic, wd))
    ic = {(tagname[t], f): float(np.mean([x[1] for x in v])) for (t, f), v in cell.items() if t in tagname}
    wf = {}
    for name in tagname.values():
        wf[name] = {"mcc": {f: mcc[(name, f)] for f in ("A", "B", "main")},
                    "rank_ic": {f: ic[(name, f)] for f in ("A", "B", "main")}}
        wf[name]["mcc"]["avg_oos"] = float(np.mean([mcc[(name, f)] for f in OOS]))
        wf[name]["rank_ic"]["avg_oos"] = float(np.mean([ic[(name, f)] for f in OOS]))
    out["walk_forward"] = wf

    # --- MMGT vs MMGT-base within-day gain, seed-matched, out-of-sample periods only
    g = [v for f in OOS for v in extra["wd_gain_by_period"][f]["per_seed"].values()]
    out["mmgt_vs_base_wd_oos"] = {"n": len(g), "pos": int(sum(x > 0 for x in g)),
                                  "mean": float(np.mean(g)),
                                  "sign_p": float(stats.binomtest(int(sum(x > 0 for x in g)), len(g)).pvalue),
                                  "t_p": float(stats.ttest_1samp(g, 0).pvalue)}

    # --- MMGT-D2 vs MMGT, seed-matched, out-of-sample periods only
    res = {}
    for k, col in (("rank_ic", 1), ("within_day", 2)):
        diffs = []
        for f in OOS:
            a = {s: x for s, *x in [(r[0], r[1], r[2]) for r in cell[("D2_final", f)]]}
            b = {s: x for s, *x in [(r[0], r[1], r[2]) for r in cell[("R_mmgt", f)]]}
            for s in sorted(set(a) & set(b)):
                diffs.append(a[s][col - 1] - b[s][col - 1])
        diffs = np.array(diffs)
        res[k] = {"n": len(diffs), "pos": int((diffs > 0).sum()), "mean": float(diffs.mean()),
                  "t_p": float(stats.ttest_1samp(diffs, 0).pvalue),
                  "sign_p": float(stats.binomtest(int((diffs > 0).sum()), len(diffs)).pvalue)}
    out["d2_vs_mmgt_oos"] = res

    # --- within-day MCC vs Rank IC agreement
    def agree(keys):
        x = [np.mean([v[1] for v in cell[k]]) for k in keys]
        y = [np.mean([v[2] for v in cell[k]]) for k in keys]
        r, p = stats.pearsonr(x, y)
        return {"n_cells": len(keys), "r": float(r), "p": float(p)}

    def agree_runs(keys):
        x, y = [], []
        for k in keys:
            v = np.array([(a[1], a[2]) for a in cell[k]])
            if len(v) > 1:
                x += list(v[:, 0] - v[:, 0].mean()); y += list(v[:, 1] - v[:, 1].mean())
        r, p = stats.pearsonr(x, y)
        return {"n_runs": len(x), "r": float(r), "p": float(p)}

    allk = sorted(cell)
    oosk = [k for k in allk if k[1] in OOS]
    out["agreement"] = {"cells_all": agree(allk), "cells_oos": agree(oosk),
                        "runs_demeaned_all": agree_runs(allk), "runs_demeaned_oos": agree_runs(oosk),
                        "models": sorted(set(k[0] for k in allk)),
                        "cells_oos_list": [list(k) for k in oosk]}

    # --- market-direction oracle (src/utils/metrics.market_direction_oracle rule:
    # predict every labelled stock of a day as the day's majority label, ties up)
    # per quarter of 2014-2015 and per test period: is the evaluation problem
    # specific to the standard test period?
    import sys
    sys.path.insert(0, os.path.dirname(H))
    from sklearn.metrics import matthews_corrcoef
    from src.data.acl18 import load_acl18
    d = load_acl18(window=10, fold="main")
    y, dates, split = d["y"], np.array(d["dates"]), d["split"]

    def oracle(idx):
        ys, ps = [], []
        for i in idx:
            lab = y[i][y[i] >= 0]
            if len(lab):
                ys += list(lab); ps += [int(lab.mean() >= 0.5)] * len(lab)
        return float(matthews_corrcoef(ys, ps))

    lab = np.where(split != "warmup")[0]
    q = pd.PeriodIndex(pd.to_datetime(dates[lab]), freq="Q").astype(str)
    out["oracle"] = {"per_quarter": {qq: oracle(lab[q == qq]) for qq in sorted(set(q))},
                     "test_main": oracle(np.where(split == "test")[0])}
    for f in ("A", "B"):
        dd = load_acl18(window=10, fold=f)
        y, split = dd["y"], dd["split"]
        out["oracle"][f"test_{f}"] = oracle(np.where(split == "test")[0])
    print("oracle:", {k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()})
                      for k, v in out["oracle"].items()})

    json.dump(out, open(os.path.join(H, "rev4_oos.json"), "w"), indent=1)
    for name, v in wf.items():
        print(f"{name:10} MCC A {v['mcc']['A']:.3f} main {v['mcc']['main']:.3f} avg {v['mcc']['avg_oos']:.3f} | B {v['mcc']['B']:.3f}"
              f"   IC A {v['rank_ic']['A']:+.3f} main {v['rank_ic']['main']:+.3f} avg {v['rank_ic']['avg_oos']:+.3f} | B {v['rank_ic']['B']:+.3f}")
    print("MMGT vs base wd (oos):", out["mmgt_vs_base_wd_oos"])
    print("D2 vs MMGT (oos):", res)
    print("agreement:", {k: v for k, v in out["agreement"].items() if k != "cells_oos_list"})


if __name__ == "__main__":
    main()
