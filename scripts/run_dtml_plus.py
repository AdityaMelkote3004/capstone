"""Experiment 11 stage C -- DTML aimed at stock discrimination, not day-calling.

Two additions, each switchable so they enter the ladder one at a time:

  --extra base|peer   Exp 10's static node features added to every stock's
                      context (volatility, EDGAR fundamentals, tweet counts; with
                      `peer`, also the peer/sector block). In Exp 10 these were the
                      only change that moved within-day direction MCC.
  --rank-alpha A      a within-day pairwise ranking term (RankNet: softplus of the
                      score gap for every up/down pair on the same day). BCE alone
                      can be satisfied by calling the day's direction; the ranking
                      term is invariant to a shift applied to every stock on a day,
                      so it can only be reduced by ordering stocks correctly.

Selection, test locking and reporting follow run_dtml.py exactly. Test is never
forwarded without --eval-test.
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.run_dtml import metrics
from src.data.acl18 import load_acl18
from src.data.acl18_extra import extra_features
from src.models.dtml import DTML
from src.utils.seed import set_seed

OUT_DIR = "results/experiment11_dtml"


def pairwise_rank_loss(logits, y):
    pos, neg = logits[y == 1], logits[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return logits.sum() * 0.0
    return nn.functional.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean()


def run_split(model, data, extra, idx, optimizer=None, reg=1.0, rank_alpha=0.0):
    train = optimizer is not None
    model.train(train)
    order = list(idx)
    if train:
        random.shuffle(order)
    probs, ys, days = [], [], []
    for i in order:
        valid = torch.from_numpy(data["window_ok"][i])
        if valid.sum() < 2:
            continue
        x = torch.from_numpy(data["x"][i])
        m = torch.from_numpy(data["market"][i])
        e = torch.from_numpy(extra[i]) if extra is not None else None
        y = torch.from_numpy(data["y"][i])
        lab = y >= 0
        with torch.set_grad_enabled(train):
            logits = model(x, valid, m, e)
            if train and lab.any():
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[lab], y[lab].float()) + reg * model.selective_penalty()
                if rank_alpha:
                    loss = loss + rank_alpha * pairwise_rank_loss(logits[lab], y[lab])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        probs.append(torch.sigmoid(logits.detach())[lab].numpy())
        ys.append(y[lab].numpy())
        days.extend([data["dates"][i]] * int(lab.sum()))
    return np.concatenate(probs), np.concatenate(ys), np.array(days)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--select", default="mcc", choices=["mcc", "within_day_mcc", "auc"])
    ap.add_argument("--ablate", default="none", choices=["none", "sa", "mc"])
    ap.add_argument("--extra", default="none", choices=["none", "base", "peer"])
    ap.add_argument("--rank-alpha", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="plus")
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()

    data = load_acl18(window=args.window)
    extra = None if args.extra == "none" else extra_features(data, peer=args.extra == "peer")
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}
    out_csv = os.path.join(OUT_DIR, "plus_final.csv" if args.eval_test else "plus_search.csv")

    for seed in args.seeds:
        set_seed(seed)
        t0 = time.time()
        model = DTML(n_stocks=len(data["tickers"]), hidden=args.hidden, heads=args.heads,
                     beta=args.beta, dropout=args.dropout,
                     use_sa=args.ablate != "sa", use_mc=args.ablate != "mc",
                     extra_dim=0 if extra is None else extra.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, state, stale, best_epoch = -9.0, None, 0, 0
        for epoch in range(args.epochs):
            run_split(model, data, extra, idx["train"], opt, args.reg, args.rank_alpha)
            vm = metrics(*run_split(model, data, extra, idx["val"]))
            if vm[args.select] > best:
                best, stale, best_epoch, best_val = vm[args.select], 0, epoch, vm
                state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(state)
        row = {"tag": args.tag, "seed": seed, "window": args.window, "beta": args.beta,
               "hidden": args.hidden, "lr": args.lr, "ablate": args.ablate,
               "extra": args.extra, "rank_alpha": args.rank_alpha,
               "select": args.select, "best_epoch": best_epoch,
               "minutes": round((time.time() - t0) / 60, 2)}
        row.update({f"val_{k}": v for k, v in best_val.items()})
        msg = (f"[{args.tag}] seed={seed} ep={best_epoch} val mcc={best_val['mcc']:+.4f} "
               f"auc={best_val['auc']:.4f} wd={best_val['within_day_mcc']:+.4f} "
               f"collapse={best_val['collapse_mcc']:+.4f}")
        if args.eval_test:
            tm = metrics(*run_split(model, data, extra, idx["test"]), boot=2000, seed=seed)
            row.update({f"test_{k}": v for k, v in tm.items()})
            msg += (f" || TEST mcc={tm['mcc']:+.4f} acc={tm['acc']:.4f} auc={tm['auc']:.4f} "
                    f"wd={tm['within_day_mcc']:+.4f} collapse={tm['collapse_mcc']:+.4f}")
        pd.DataFrame([row]).to_csv(out_csv, mode="a",
                                   header=not os.path.exists(out_csv), index=False)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
