"""Every number quoted in the paper, computed from the result CSVs.

Only leak-free results are used: profile-based models carry the "c" tag
(F1c, F2c, F3c, G2c, FEDc); F0, F4, S4 and the baselines never used the profile.
Writes paper/numbers.json and prints a summary.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results", "experiment11_dtml")

d = pd.concat([pd.read_csv(os.path.join(R, f)) for f in
               ["final.csv", "plus_final.csv", "graph_final.csv",
                "baselines_final.csv", "subset_final.csv"]], ignore_index=True)
fed_path = os.path.join(R, "federated_final.csv")
if os.path.exists(fed_path):
    d = pd.concat([d, pd.read_csv(fed_path)], ignore_index=True)

# F1 was not logged for Experiment 11, but each run logged accuracy and the share
# of "up" predictions, which with the test set's up rate (1908 of 3720) fix the
# confusion matrix exactly; the MCC recomputed from it matches the logged MCC to
# 1e-15 on every run. Macro-F1 averages the F1 of the "up" and "down" classes, so
# predicting one class more often cannot inflate it (always-up: F1(up) 0.678,
# macro-F1 0.339).
N_TEST, P_UP = 3720, 1908 / 3720


def confusion(acc, q):
    tp = (acc * N_TEST - N_TEST + q * N_TEST + P_UP * N_TEST) / 2
    fp, fn = q * N_TEST - tp, P_UP * N_TEST - tp
    return tp, fp, fn, N_TEST - tp - fp - fn


def add_f1(frame, q):
    tp, fp, fn, tn = confusion(frame["test_acc"].to_numpy(), q)
    frame["test_f1_up"] = 2 * tp / (2 * tp + fp + fn)
    frame["test_macro_f1"] = (frame["test_f1_up"] + 2 * tn / (2 * tn + fn + fp)) / 2
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert np.nanmax(np.abs(mcc - frame["test_mcc"].to_numpy())) < 1e-9, "confusion rebuild failed"
    return frame


d = add_f1(d, d["test_pos_rate_pred"].to_numpy())

e6 = pd.read_csv(os.path.join(ROOT, "results", "experiment6_dense_graph", "results.csv"))
e6 = e6[(e6.model == "RGCN") & (e6.edges == "sector+correlation")].copy()
# Exp 6 logged accuracy and F1(up) but not the prediction rate; solve for it.
e6_q = (e6.accuracy - 1 + P_UP - e6.f1 * P_UP) / (e6.f1 - 1)
e6 = add_f1(e6.rename(columns={"accuracy": "test_acc", "mcc": "test_mcc"}), e6_q.to_numpy())
e6 = e6.rename(columns={"test_acc": "accuracy", "test_mcc": "mcc"})
EXP1 = np.array([0.0027, -0.0009, 0.0011, 0.0181, -0.0125])   # Exp 1 write-up, 5 seeds

TAGS = {"lr": "B_lr", "mlp": "B_mlp", "lstm": "B_lstm", "alstm": "B_alstm",
        "dtml": "F0_dtml", "dtml_nosa": "F4_nosa", "dtml_7f": "S4_ret_trend",
        "mmgt": "F1c", "mmgt_rank": "F2c", "mmgt_rank_nomc": "F3c",
        "mmgt_graphbias": "G2c", "mmgt_fed": "FEDc"}


def col(tag, c="test_mcc"):
    return d[d.tag == tag].sort_values("seed")[c].to_numpy()


out = {"models": {}, "tests": {}}
for name, tag in TAGS.items():
    x = d[d.tag == tag].sort_values("seed")
    if not len(x):
        continue
    m = {"n": len(x)}
    for k in ["mcc", "acc", "auc", "within_day_mcc", "collapse_mcc", "f1_up", "macro_f1"]:
        v = x[f"test_{k}"].to_numpy()
        m[k] = float(v.mean())
        m[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
    m["seeds"] = [round(float(v), 4) for v in x.test_mcc]
    out["models"][name] = m
out["models"]["rgcn_exp6"] = {"n": len(e6), "mcc": float(e6.mcc.mean()),
                              "mcc_sd": float(e6.mcc.std()),
                              "acc": float(e6.accuracy.mean()), "auc": float(e6.auc.mean()),
                              "macro_f1": float(e6.test_macro_f1.mean()),
                              "f1_up": float(e6.test_f1_up.mean()),
                              "within_day_mcc": float(e6.within_day_mcc.mean())}
out["models"]["always_up"] = {"n": 1, "mcc": 0.0, "acc": P_UP, "auc": 0.5,
                              "f1_up": 2 * P_UP / (1 + P_UP), "macro_f1": P_UP / (1 + P_UP)}
out["models"]["tweetlstm_exp1"] = {"n": 5, "mcc": float(EXP1.mean()),
                                   "mcc_sd": float(EXP1.std(ddof=1))}

T = out["tests"]
baselines = {"lr": col("B_lr"), "mlp": col("B_mlp"), "lstm": col("B_lstm"),
             "alstm": col("B_alstm"), "rgcn_exp6": e6.mcc.to_numpy(), "tweetlstm_exp1": EXP1}
for model in ["mmgt", "dtml", "mmgt_rank", "mmgt_graphbias", "dtml_7f", "mmgt_fed"]:
    if model not in out["models"]:
        continue
    y = col(TAGS[model])
    for b, x in baselines.items():
        if len(x) == 1:
            p = stats.ttest_1samp(y, x[0]).pvalue
        else:
            p = stats.ttest_ind(y, x, equal_var=False).pvalue
        T[f"{model}_vs_{b}"] = float(p)


def paired(a, b, c="test_mcc"):
    x, y = col(TAGS[a], c), col(TAGS[b], c)
    n = min(len(x), len(y))
    return float(stats.ttest_rel(x[:n], y[:n]).pvalue), int((x[:n] > y[:n]).sum()), n


for b in ["lr", "mlp", "lstm", "alstm"]:
    y, x = col("F2c", "test_macro_f1"), col(TAGS[b], "test_macro_f1")
    T[f"mmgt_rank_vs_{b}_macro_f1"] = float(
        stats.ttest_1samp(y, x[0]).pvalue if len(x) == 1 else stats.ttest_ind(y, x, equal_var=False).pvalue)
T["mmgt_rank_vs_rgcn_exp6_macro_f1"] = float(stats.ttest_ind(
    col("F2c", "test_macro_f1"), e6.test_macro_f1.to_numpy(), equal_var=False).pvalue)
for a, b in [("mmgt", "dtml"), ("mmgt_graphbias", "mmgt"), ("mmgt_rank", "mmgt"),
             ("dtml_7f", "dtml"), ("mmgt_fed", "mmgt")]:
    if a in out["models"] and b in out["models"]:
        for c, short in [("test_mcc", "pooled"), ("test_within_day_mcc", "wd")]:
            p, wins, n = paired(a, b, c)
            T[f"{a}_vs_{b}_{short}_paired_p"] = p
            T[f"{a}_vs_{b}_{short}_wins"] = f"{wins}/{n}"
for model in ["mmgt", "dtml", "lstm", "alstm", "mlp", "mmgt_rank", "mmgt_graphbias"]:
    v = col(TAGS[model], "test_within_day_mcc")
    T[f"{model}_wd_vs0_p"] = float(stats.ttest_1samp(v, 0).pvalue)
T["mmgt_seeds_above_best_baseline"] = int((col("F1c") > max(
    out["models"]["mlp"]["mcc"], out["models"]["rgcn_exp6"]["mcc"])).sum()) if "mmgt" in out["models"] else None
if "mmgt_fed" in out["models"]:
    f = d[d.tag == "FEDc"].sort_values("seed")
    T["fed_best_round_1idx"] = [int(r) + 1 for r in f.best_round]
    T["fed_rounds_run"] = [int(r) for r in f.rounds_run]
    T["mmgt_same_seeds_as_fed_mcc"] = float(col("F1c")[f.seed.to_numpy()].mean())
    T["mmgt_same_seeds_as_fed_wd"] = float(col("F1c", "test_within_day_mcc")[f.seed.to_numpy()].mean())

json.dump(out, open(os.path.join(HERE, "numbers.json"), "w"), indent=1)
for k, v in out["models"].items():
    print(f"{k:16} n={v['n']:2}  mcc={v['mcc']:+.4f}"
          + (f" ±{v['mcc_sd']:.4f}" if v.get("mcc_sd") else "")
          + (f"  acc={v['acc']:.4f} auc={v['auc']:.4f}" if "acc" in v else "")
          + (f"  wd={v['within_day_mcc']:+.4f}" if "within_day_mcc" in v else "")
          + (f"±{v['within_day_mcc_sd']:.4f}" if v.get("within_day_mcc_sd") else "")
          + (f"  collapse={v['collapse_mcc']:+.4f}" if "collapse_mcc" in v else ""))
for k, v in T.items():
    print(f"  {k}: {v:.2e}" if isinstance(v, float) and v < 0.01 else f"  {k}: {v}")
