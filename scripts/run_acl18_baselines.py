"""Experiment 11 baselines -- logistic regression and MLP on the ACL18 inputs.

Both use the flattened 10-day x 11-feature price window, standardised with
train-split statistics. (LSTM and attention-LSTM baselines run through
scripts/run_dtml.py with --ablate all / --ablate samc.)

Revision after review: hyperparameters are tunable (--C for LR; --hidden, --lr
for the MLP) so the baselines get a validation search like the main models;
--eval-test locks test as elsewhere; --save-preds stores predictions for every
window-valid stock for date-level statistics and the backtest; --fold selects a
walk-forward split.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from scripts.run_dtml import metrics
from src.data.acl18 import load_acl18
from src.utils.seed import set_seed

OUT_DIR = "results/experiment11_dtml"


def flatten(data, idx, labelled_only=True):
    xs, ys, days, stocks = [], [], [], []
    for i in idx:
        ok = data["window_ok"][i] & ((data["y"][i] >= 0) if labelled_only else True)
        s = np.where(ok)[0]
        xs.append(data["x"][i][ok].reshape(len(s), -1))
        ys.append(data["y"][i][ok])
        days.extend([i] * len(s))
        stocks.append(s)
    return np.concatenate(xs), np.concatenate(ys), np.array(days), np.concatenate(stocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "mlp"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fold", default="main", choices=["main", "A", "B"])
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()
    tag = args.tag or f"B_{args.model}"

    data = load_acl18(window=args.window, fold=args.fold)
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}
    xtr, ytr, _, _ = flatten(data, idx["train"])
    xva, yva, dva, _ = flatten(data, idx["val"])
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    norm = lambda x: np.clip((x - mu) / sd, -10, 10).astype(np.float32)
    xtr, xva = norm(xtr), norm(xva)
    dates = np.array(data["dates"])

    out = os.path.join(OUT_DIR, args.out_prefix + ("baselines_final.csv" if args.eval_test
                                                  else "baselines_search.csv"))
    for seed in args.seeds:
        set_seed(seed)
        if args.model == "lr":
            clf = LogisticRegression(C=args.C, max_iter=2000, random_state=seed).fit(xtr, ytr)
            predict = lambda x: clf.predict_proba(x)[:, 1]
            best_epoch = -1
        else:
            net = nn.Sequential(nn.Linear(xtr.shape[1], args.hidden), nn.ReLU(), nn.Dropout(0.15),
                                nn.Linear(args.hidden, args.hidden // 2), nn.ReLU(),
                                nn.Linear(args.hidden // 2, 1))
            opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
            Xtr, Ytr, Xva = torch.from_numpy(xtr), torch.from_numpy(ytr).float(), torch.from_numpy(xva)
            best, state, stale, best_epoch = -9.0, None, 0, 0
            for epoch in range(200):
                net.train()
                perm = torch.randperm(len(Xtr))
                for b in range(0, len(perm), 256):
                    j = perm[b:b + 256]
                    loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[j]).squeeze(-1), Ytr[j])
                    opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(net(Xva).squeeze(-1)).numpy()
                m = metrics(pv, yva, dates[dva])["mcc"]
                if m > best:
                    best, stale, best_epoch = m, 0, epoch
                    state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 30:
                        break
            net.load_state_dict(state); net.eval()

            def predict(x):
                with torch.no_grad():
                    return torch.sigmoid(net(torch.from_numpy(x)).squeeze(-1)).numpy()

        vm = metrics(predict(xva), yva, dates[dva])
        row = {"tag": tag, "seed": seed, "fold": args.fold, "C": args.C,
               "hidden": args.hidden, "lr": args.lr, "best_epoch": best_epoch}
        row.update({f"val_{k}": v for k, v in vm.items()})
        msg = f"[{tag}] seed={seed} val mcc={vm['mcc']:+.4f}"
        if args.eval_test:
            xte, yte, dte, _ = flatten(data, idx["test"])
            tm = metrics(predict(norm(xte)), yte, dates[dte], boot=2000, seed=seed)
            row.update({f"test_{k}": v for k, v in tm.items()})
            msg += f" || TEST mcc={tm['mcc']:+.4f} acc={tm['acc']:.4f} wd={tm['within_day_mcc']:+.4f}"
            if args.save_preds:
                from src.eval.predictions import save
                xa, ya, da, sa = flatten(data, idx["test"], labelled_only=False)
                save(os.path.join(OUT_DIR, "preds", f"{tag}_{args.fold}_s{seed}.npz"),
                     {"day": da, "stock": sa, "prob": predict(norm(xa)), "label": ya,
                      "ret": data["ret"][da, sa]}, data["dates"])
        pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
