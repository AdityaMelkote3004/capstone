"""Experiment 13 (a): MMGT-D2 ablation table (disclosed post-hoc; see PREREGISTRATION.md).

Rows: MMGT (R_mmgt), two-MMGT ensemble (Exp 12b capacity control), the four MMGT-D2
configurations (D2_final from exp12b_final.csv; the three others from exp13_final.csv).
Per row: mean +- s.d. over ten runs of test MCC, within-day MCC, day-collapse MCC, and
Rank IC (from saved predictions, all window-valid stocks); Welch p vs MMGT for MCC and
Rank IC. Writes paper/exp13_ablation.json.
"""
import json, os

import numpy as np
import pandas as pd
from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(H), "results", "experiment11_dtml")
P = os.path.join(R, "preds")


def rank_ic(tag, seed):
    z = np.load(os.path.join(P, f"{tag}_main_s{seed}.npz"))
    day, prob, ret = z["day"], z["prob"], z["ret"]
    ics = []
    for d in np.unique(day):
        k = (day == d) & np.isfinite(ret)
        if k.sum() > 2:
            ics.append(stats.spearmanr(prob[k], ret[k])[0])
    return float(np.mean(ics))


def main():
    mm = pd.read_csv(os.path.join(R, "rev_plus_final.csv"))
    mm = mm[(mm.tag == "R_mmgt") & (mm.fold == "main")]
    d2 = pd.read_csv(os.path.join(R, "exp12b_final.csv"))
    d2 = d2[(d2.tag == "D2_final") & (d2.fold == "main")]
    ab = pd.read_csv(os.path.join(R, "exp13_final.csv"))
    rows = {"MMGT": ("R_mmgt", mm), "D2_final": ("D2_final", d2)}
    for tag in ("D2_la0.5", "D2_la1", "D2adv_la0.5"):
        rows[tag] = (tag, ab[ab.tag == tag])
    out = {}
    ic_mmgt = np.array([rank_ic("R_mmgt", s) for s in sorted(mm.seed)])
    for name, (tag, df) in rows.items():
        df = df.sort_values("seed")
        ics = np.array([rank_ic(tag, s) for s in df.seed])
        r = {"n": int(len(df))}
        for col in ("test_mcc", "test_acc", "test_within_day_mcc", "test_collapse_mcc"):
            r[col] = [float(df[col].mean()), float(df[col].std())]
        r["rank_ic"] = [float(ics.mean()), float(ics.std(ddof=1))]
        if name != "MMGT":
            r["p_mcc_vs_mmgt"] = float(stats.ttest_ind(df.test_mcc, mm.test_mcc, equal_var=False).pvalue)
            r["p_ic_vs_mmgt"] = float(stats.ttest_ind(ics, ic_mmgt, equal_var=False).pvalue)
        out[name] = r
    cap = json.load(open(os.path.join(H, "exp12b_capacity_control.json")))
    out["ensemble_2xMMGT"] = {"n": len(cap["pairs"]), "test_mcc": [cap["mcc_mean"], float(np.std([p[0] for p in cap["pairs"]], ddof=1))],
                              "rank_ic": [cap["ic_mean"], float(np.std([p[1] for p in cap["pairs"]], ddof=1))]}
    json.dump(out, open(os.path.join(H, "exp13_ablation.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:16}", {kk: (round(vv[0], 4) if isinstance(vv, list) else round(vv, 4)) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
