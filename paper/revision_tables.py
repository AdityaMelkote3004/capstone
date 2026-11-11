"""Tables for the revised paper from the rev_* CSVs and saved predictions.
Macro-F1 is computed directly from saved predictions (labelled rows, threshold 0.5).
Writes paper/revision_tables.json."""
import glob, json, os, re
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(os.path.dirname(H), "results", "experiment11_dtml")
TAGS = {"MMGT": "R_mmgt", "MMGT-base": "R_base", "Profile only": "R_profile", "LR": "R_lr", "MLP": "R_mlp",
        "LSTM": "R_lstm", "ALSTM": "R_alstm", "Federated": "FEDlong"}
frames = []
for f in ["rev_final.csv", "rev_plus_final.csv", "rev_baselines_final.csv", "rev_federated_final.csv"]:
    x = pd.read_csv(os.path.join(R, f))
    if "fold" not in x: x["fold"] = "main"
    frames.append(x[["tag", "seed", "fold", "test_mcc", "test_acc", "test_auc", "test_within_day_mcc", "test_collapse_mcc"]])
df = pd.concat(frames)
def f1(tag, fold, seed):
    z = np.load(os.path.join(R, "preds", f"{tag}_{fold}_s{seed}.npz")); m = z["label"] >= 0
    return f1_score(z["label"][m], (z["prob"][m] >= 0.5).astype(int), average="macro")
df["test_f1"] = [f1(t, fo, s) for t, fo, s in zip(df.tag, df.fold, df.seed)]
out = {}
for fold in ["main", "A", "B"]:
    out[fold] = {}
    mm = df[(df.tag == "R_mmgt") & (df.fold == fold)].test_mcc
    for name, tag in TAGS.items():
        x = df[(df.tag == tag) & (df.fold == fold)]
        if x.empty: continue
        row = {"n": len(x)}
        for c in ["test_mcc", "test_acc", "test_f1", "test_auc", "test_within_day_mcc", "test_collapse_mcc"]:
            row[c] = [float(x[c].mean()), float(x[c].std(ddof=1)) if len(x) > 1 else 0.0]
        if name != "MMGT" and len(x) > 1:
            row["welch_p_vs_mmgt"] = float(stats.ttest_ind(mm, x.test_mcc, equal_var=False).pvalue)
        if name != "MMGT" and len(x) == 1:
            row["one_sample_p_vs_mmgt"] = float(stats.ttest_1samp(mm, x.test_mcc.iloc[0]).pvalue)
        out[fold][name] = row
json.dump(out, open(os.path.join(H, "revision_tables.json"), "w"), indent=1)
for fold, rows in out.items():
    print("==", fold)
    for n, r in rows.items():
        print(f"{n:12} n={r['n']:2} " + " ".join(f"{k[5:]}={v[0]:.3f}±{v[1]:.3f}" for k, v in r.items() if k.startswith("test_"))
              + "".join(f" p={v:.2g}" for k, v in r.items() if k.endswith("mmgt")))
