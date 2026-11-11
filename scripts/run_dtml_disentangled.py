"""Experiment 12 -- MMGT-D, the disentangled MMGT (see PREREGISTRATION.md).

Same inputs, backbone settings, selection rule and test locking as MMGT
(run_dtml_plus.py with --extra peer --rank-alpha 1). Added losses:
  lambda_m  * BCE(m_t, market_up_t)   market head predicts the day's direction
  lambda_adv* BCE(adv_t, market_up_t) adversary (gradient reversed into the stock branch)
market_up_t = mean target-day return over WINDOW-VALID stocks > 0. It is a training
label only and is computed only for training days.
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
from scripts.run_dtml_plus import pairwise_rank_loss
from src.data.acl18 import load_acl18
from src.data.acl18_extra import extra_features
from src.models.dtml_disentangled import DTMLDisentangled
from src.utils.seed import set_seed

OUT_DIR = "results/experiment11_dtml"


def run_split(model, data, extra, idx, opt=None, args=None):
    train = opt is not None
    model.train(train)
    order = list(idx)
    if train:
        random.shuffle(order)
    probs, ys, days = [], [], []
    bce = nn.functional.binary_cross_entropy_with_logits
    for i in order:
        valid = torch.from_numpy(data["window_ok"][i])
        if valid.sum() < 2:
            continue
        y = torch.from_numpy(data["y"][i])
        lab = y >= 0
        with torch.set_grad_enabled(train):
            logits = model(torch.from_numpy(data["x"][i]), valid,
                           torch.from_numpy(data["market"][i]), torch.from_numpy(extra[i]))
            if train and lab.any():
                r = data["ret"][i][data["window_ok"][i]]
                mkt = torch.tensor(float(np.nanmean(r) > 0))
                loss = (bce(logits[lab], y[lab].float()) + args.reg * model.selective_penalty()
                        + args.lambda_m * bce(model.parts["m"][0], mkt)
                        + args.rank_alpha * pairwise_rank_loss(logits[lab], y[lab]))
                if model.parts["adv"] is not None:
                    loss = loss + args.lambda_adv * bce(model.parts["adv"], mkt)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        probs.append(torch.sigmoid(logits.detach())[lab].numpy())
        ys.append(y[lab].numpy())
        days.extend([data["dates"][i]] * int(lab.sum()))
    return np.concatenate(probs), np.concatenate(ys), np.array(days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["struct", "beta"], default="struct")
    ap.add_argument("--adv", action="store_true")
    ap.add_argument("--lambda-m", type=float, default=1.0)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--rank-alpha", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="D")
    ap.add_argument("--fold", default="main", choices=["main", "A", "B"])
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--out-prefix", default="exp12_")
    args = ap.parse_args()

    data = load_acl18(window=args.window, fold=args.fold)
    extra = extra_features(data, peer=True)
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}
    out_csv = os.path.join(OUT_DIR, args.out_prefix + ("final.csv" if args.eval_test else "search.csv"))

    for seed in args.seeds:
        set_seed(seed)
        t0 = time.time()
        model = DTMLDisentangled(n_stocks=len(data["tickers"]), hidden=args.hidden, heads=4,
                                 dropout=args.dropout, extra_dim=extra.shape[-1],
                                 mode=args.mode, adversary=args.adv)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, state, stale, best_epoch = -9.0, None, 0, 0
        for epoch in range(args.epochs):
            run_split(model, data, extra, idx["train"], opt, args)
            vm = metrics(*run_split(model, data, extra, idx["val"]))
            if vm["mcc"] > best:
                best, stale, best_epoch, best_val = vm["mcc"], 0, epoch, vm
                state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(state)
        row = {"tag": args.tag, "seed": seed, "fold": args.fold, "mode": args.mode,
               "adv": args.adv, "lambda_m": args.lambda_m, "best_epoch": best_epoch,
               "minutes": round((time.time() - t0) / 60, 2)}
        row.update({f"val_{k}": v for k, v in best_val.items()})
        msg = (f"[{args.tag}] seed={seed} fold={args.fold} ep={best_epoch} "
               f"val mcc={best_val['mcc']:+.4f} wd={best_val['within_day_mcc']:+.4f}")
        if args.eval_test:
            tm = metrics(*run_split(model, data, extra, idx["test"]), boot=2000, seed=seed)
            row.update({f"test_{k}": v for k, v in tm.items()})
            if args.save_preds:
                from src.eval.predictions import collect, save
                save(os.path.join(OUT_DIR, "preds", f"{args.tag}_{args.fold}_s{seed}.npz"),
                     collect(model, data, idx["test"], extra), data["dates"])
            msg += (f" || TEST mcc={tm['mcc']:+.4f} acc={tm['acc']:.4f} "
                    f"wd={tm['within_day_mcc']:+.4f} collapse={tm['collapse_mcc']:+.4f}")
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
