"""Experiment 12 endpoint evaluation: MMGT-D (D_final) vs MMGT (R_mmgt) and others.
Writes paper/exp12_eval.json."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, matthews_corrcoef
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(os.path.dirname(H), "results", "experiment11_dtml")
fin = json.load(open(os.path.join(H, "revision_finance.json")))
ic = {}
for tag, fold, seed, i, w in fin["agreement"]["runs"]:
    ic.setdefault((tag, fold), {})[seed] = i
d = pd.read_csv(os.path.join(R, "exp12_final.csv"))
m = pd.read_csv(os.path.join(R, "rev_plus_final.csv")); m = m[m.tag == "R_mmgt"]
base = pd.read_csv(os.path.join(R, "rev_final.csv")); base = base[base.tag == "R_base"]
cols = ["test_mcc", "test_acc", "test_auc", "test_within_day_mcc", "test_collapse_mcc"]
out = {}
exec(open(os.path.join(H, "revision_extra.py")).read().split('out = {"wd_bootstrap"')[0].replace(
    'H = os.path.dirname(os.path.abspath(__file__))', f'H = r"{H}"'))
for fold in ["main", "A", "B"]:
    a, b, c = d[d.fold == fold].sort_values("seed"), m[m.fold == fold].sort_values("seed"), base[base.fold == fold].sort_values("seed")
    row = {"n": len(a)}
    for col in cols:
        row[col] = {"D": [a[col].mean(), a[col].std()], "MMGT": [b[col].mean(), b[col].std()], "base": [c[col].mean(), c[col].std()],
                    "welch_p_D_vs_MMGT": stats.ttest_ind(a[col], b[col], equal_var=False).pvalue}
    ia, ib = np.array(list(ic[("D_final", fold)].values())), np.array(list(ic[("R_mmgt", fold)].values()))
    row["rank_ic"] = {"D": [ia.mean(), ia.std(ddof=1)], "MMGT": [ib.mean(), ib.std(ddof=1)],
                      "welch_p": stats.ttest_ind(ia, ib, equal_var=False).pvalue}
    seeds = range(10) if fold == "main" else range(5)
    rd, rm = load("D_final", fold, seeds), load("R_mmgt", fold, seeds)
    row["wd_bootstrap_D_minus_MMGT"] = boot_wd(rd, rm)
    # F1 from predictions
    f1 = lambda rr: np.mean([f1_score(r["label"][r["label"] >= 0], (r["prob"][r["label"] >= 0] >= .5).astype(int), average="macro") for r in rr.values()])
    row["macro_f1"] = {"D": f1(rd), "MMGT": f1(rm)}
    # components: market-only (sigmoid m) and stock-only (e) diagnostics
    def comp(rr):
        res = {"market_only_mcc": [], "excess_wd": [], "excess_ic": []}
        for r in rr.values():
            lab = r["label"] >= 0
            res["market_only_mcc"].append(matthews_corrcoef(r["label"][lab], (r["m"][lab] >= 0).astype(int)))
            dd, y, e = r["day"][lab], r["label"][lab], r["e"][lab]
            res["excess_wd"].append(np.mean([matthews_corrcoef(y[dd == t], (e[dd == t] >= np.median(e[dd == t])).astype(int))
                                             for t in np.unique(dd) if len(set(y[dd == t])) > 1]))
            res["excess_ic"].append(np.mean([stats.spearmanr(r["e"][r["day"] == t], r["ret"][r["day"] == t])[0] for t in np.unique(r["day"])]))
        return {k: float(np.mean(v)) for k, v in res.items()}
    full = {sd: dict(np.load(os.path.join(P, f"D_final_{fold}_s{sd}.npz"))) for sd in seeds}
    row["components"] = comp(full)
    out[fold] = row
if True:
    exec(open(os.path.join(H, "revision_analysis.py")).read().split("def main")[0].replace(
        'HERE = os.path.dirname(os.path.abspath(__file__))', f'HERE = r"{H}"'))
    out["main"]["pooled_bootstrap_D_minus_MMGT"] = bootstrap(load("D_final", "main"), load("R_mmgt", "main"))
    for name, tag in [("MLP", "R_mlp"), ("LSTM", "R_lstm"), ("ALSTM", "R_alstm"), ("LR", "R_lr")]:
        out["main"][f"pooled_bootstrap_D_minus_{name}"] = bootstrap(load("D_final", "main"), load(tag, "main"))
json.dump(out, open(os.path.join(H, "exp12_eval.json"), "w"), indent=1, default=float)
for fold, row in out.items():
    print("=====", fold, "n =", row["n"])
    for col in cols:
        v = row[col]; print(f"  {col:22} D {v['D'][0]:+.4f}±{v['D'][1]:.4f}  MMGT {v['MMGT'][0]:+.4f}  base {v['base'][0]:+.4f}  p={v['welch_p_D_vs_MMGT']:.3g}")
    v = row["rank_ic"]; print(f"  rank IC                D {v['D'][0]:+.4f}±{v['D'][1]:.4f}  MMGT {v['MMGT'][0]:+.4f}  p={v['welch_p']:.3g}")
    print("  macro-F1", {k: round(x, 3) for k, x in row["macro_f1"].items()})
    print("  wd day-bootstrap D-MMGT", {k: round(x, 4) if isinstance(x, float) else x for k, x in row["wd_bootstrap_D_minus_MMGT"].items()})
    print("  components", {k: round(x, 4) for k, x in row["components"].items()})
    for k, v in row.items():
        if k.startswith("pooled_bootstrap"):
            print(f"  {k}: {v['diff']:+.4f} [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p']:.3f}")
