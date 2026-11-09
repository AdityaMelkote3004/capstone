"""Experiment 11 baselines -- the Phase 1 model families on the ACL18 test set.

Phase 1 trained logistic regression, MLP and LSTM baselines on a different split
(test Aug-Dec 2015), so its numbers cannot be compared directly with Experiment
11 (test Oct-Dec 2015, n = 3,720). This script retrains the same families on
exactly the Experiment 11 inputs and split:

  LR    logistic regression on the flattened 10-day x 11-feature window
  MLP   two hidden layers (64, 32) on the same flattened window
  (LSTM and attention-LSTM run through scripts/run_dtml.py with
   --ablate all / --ablate samc, i.e. DTML with its components removed)

Inputs are standardised with train-split statistics. Neural baselines select
their checkpoint on validation MCC exactly like the main models. Baselines were
NOT hyperparameter-tuned, which favours the main models; this is disclosed.
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

OUT = "results/experiment11_dtml/baselines_final.csv"


def flatten(data, idx):
    xs, ys, days = [], [], []
    for i in idx:
        ok = data["window_ok"][i] & (data["y"][i] >= 0)
        xs.append(data["x"][i][ok].reshape(ok.sum(), -1))
        ys.append(data["y"][i][ok])
        days.extend([data["dates"][i]] * int(ok.sum()))
    return np.concatenate(xs), np.concatenate(ys), np.array(days)


def record(row):
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "mlp"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    data = load_acl18(window=args.window)
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}
    (xtr, ytr, _), (xva, yva, dva), (xte, yte, dte) = (flatten(data, idx[s])
                                                       for s in ("train", "val", "test"))
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    xtr, xva, xte = [np.clip((x - mu) / sd, -10, 10).astype(np.float32)
                     for x in (xtr, xva, xte)]

    for seed in args.seeds:
        set_seed(seed)
        if args.model == "lr":
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(xtr, ytr)
            pv, pt = clf.predict_proba(xva)[:, 1], clf.predict_proba(xte)[:, 1]
            best_epoch = -1
        else:
            net = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.15),
                                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            Xtr, Ytr = torch.from_numpy(xtr), torch.from_numpy(ytr).float()
            Xva, Xte = torch.from_numpy(xva), torch.from_numpy(xte)
            best, state, stale, best_epoch = -9.0, None, 0, 0
            for epoch in range(200):
                net.train()
                perm = torch.randperm(len(Xtr))
                for b in range(0, len(perm), 256):
                    j = perm[b:b + 256]
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        net(Xtr[j]).squeeze(-1), Ytr[j])
                    opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(net(Xva).squeeze(-1)).numpy()
                m = metrics(pv, yva, dva)["mcc"]
                if m > best:
                    best, stale, best_epoch = m, 0, epoch
                    state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 30:
                        break
            net.load_state_dict(state); net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xva).squeeze(-1)).numpy()
                pt = torch.sigmoid(net(Xte).squeeze(-1)).numpy()
        vm = metrics(pv, yva, dva)
        tm = metrics(pt, yte, dte, boot=2000, seed=seed)
        row = {"tag": f"B_{args.model}", "seed": seed, "best_epoch": best_epoch}
        row.update({f"val_{k}": v for k, v in vm.items()})
        row.update({f"test_{k}": v for k, v in tm.items()})
        record(row)
        print(f"[B_{args.model}] seed={seed} val mcc={vm['mcc']:+.4f} || TEST mcc={tm['mcc']:+.4f} "
              f"acc={tm['acc']:.4f} auc={tm['auc']:.4f} wd={tm['within_day_mcc']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
