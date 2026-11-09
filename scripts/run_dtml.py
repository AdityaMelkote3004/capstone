"""Experiment 11 -- DTML on ACL18, reproduced and then extended.

TEST IS LOCKED BY DEFAULT. Without --eval-test, a run trains, selects its
checkpoint on validation, and records validation metrics only; the test split is
never forwarded. All hyperparameter search and every iteration happens that way.
Test is touched once, for the configuration chosen on validation and its
ablations. The number of configurations tried is logged so the paper can report
it.

Every evaluated run reports, alongside DTML's own ACC/MCC:
  within_day_mcc   ranking inside each day's cross-section (stock skill)
  collapse_mcc     pooled MCC after replacing each day's predictions by that
                   day's mean probability -- i.e. what a pure day-caller scores.
                   If collapse_mcc is close to mcc, the model is timing the
                   market, not picking stocks (the failure behind Exp 6 and 7).
  mcc_ci_lo/hi     95% CI from resampling DAYS, not rows: the market component
                   of a 64-day test set has an effective n near 64, not 3,720.
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
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

from src.data.acl18 import load_acl18
from src.models.dtml import DTML
from src.utils.metrics import within_day_metrics
from src.utils.seed import set_seed

OUT_DIR = "results/experiment11_dtml"


def run_split(model, data, idx, optimizer=None, reg=1.0):
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
        y = torch.from_numpy(data["y"][i])
        labelled = y >= 0
        with torch.set_grad_enabled(train):
            logits = model(x, valid, m)
            if train and labelled.any():
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[labelled], y[labelled].float())
                loss = loss + reg * model.selective_penalty()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        p = torch.sigmoid(logits.detach())[labelled].numpy()
        probs.append(p)
        ys.append(y[labelled].numpy())
        days.extend([data["dates"][i]] * int(labelled.sum()))
    return np.concatenate(probs), np.concatenate(ys), np.array(days)


def metrics(p, y, days, boot: int = 0, seed: int = 0) -> dict:
    pred = (p >= 0.5).astype(int)
    out = {"acc": accuracy_score(y, pred), "mcc": matthews_corrcoef(y, pred),
           "auc": roc_auc_score(y, p) if len(set(y)) > 1 else 0.5,
           "n": int(len(y)), "pos_rate_pred": float(pred.mean())}
    out.update(within_day_metrics(y, p, days))
    day_mean = pd.Series(p).groupby(days).transform("mean").to_numpy()
    out["collapse_mcc"] = matthews_corrcoef(y, (day_mean >= 0.5).astype(int))
    if boot:
        rng = np.random.default_rng(seed)
        uniq = np.unique(days)
        by_day = {d: np.where(days == d)[0] for d in uniq}
        draws = []
        for _ in range(boot):
            pick = np.concatenate([by_day[d] for d in rng.choice(uniq, len(uniq))])
            draws.append(matthews_corrcoef(y[pick], pred[pick]))
        out["mcc_ci_lo"], out["mcc_ci_hi"] = np.percentile(draws, [2.5, 97.5])
    return out


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
    ap.add_argument("--select", default="mcc", choices=["mcc", "acc", "within_day_mcc"])
    ap.add_argument("--ablate", default="none", choices=["none", "ta", "sa", "mc", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-test", action="store_true")
    # Feature-subset runs (stage D) write to their own files so the column set of
    # search.csv / final.csv is never changed mid-experiment.
    ap.add_argument("--features", type=int, nargs="+", default=None,
                    help="indices into DTML's 11 features; default all")
    args = ap.parse_args()

    data = load_acl18(window=args.window)
    if args.features is not None:
        data["x"] = np.ascontiguousarray(data["x"][..., args.features])
        data["market"] = np.ascontiguousarray(data["market"][..., args.features])
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}
    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = "subset_" if args.features is not None else ""
    out_csv = os.path.join(OUT_DIR, prefix + ("final.csv" if args.eval_test else "search.csv"))

    for seed in args.seeds:
        set_seed(seed)
        model = DTML(n_stocks=len(data["tickers"]), n_features=data["x"].shape[-1],
                     hidden=args.hidden,
                     heads=args.heads, beta=args.beta, dropout=args.dropout,
                     use_ta=args.ablate not in ("ta", "all"),
                     use_sa=args.ablate not in ("sa", "all"),
                     use_mc=args.ablate not in ("mc", "all"))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, best_state, best_epoch, stale, t0 = -9.0, None, 0, 0, time.time()
        for epoch in range(args.epochs):
            run_split(model, data, idx["train"], opt, args.reg)
            vm = metrics(*run_split(model, data, idx["val"]))
            if vm[args.select] > best:
                best, best_epoch, stale = vm[args.select], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_val = vm
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        row = {"tag": args.tag, "seed": seed, "window": args.window, "beta": args.beta,
               "hidden": args.hidden, "heads": args.heads, "lr": args.lr,
               "reg": args.reg, "dropout": args.dropout, "ablate": args.ablate,
               "select": args.select, "best_epoch": best_epoch,
               **({"features": "-".join(map(str, args.features))}
                  if args.features is not None else {}),
               "minutes": round((time.time() - t0) / 60, 2)}
        row.update({f"val_{k}": v for k, v in best_val.items()})
        if args.eval_test:
            tm = metrics(*run_split(model, data, idx["test"]), boot=2000, seed=seed)
            row.update({f"test_{k}": v for k, v in tm.items()})
        pd.DataFrame([row]).to_csv(out_csv, mode="a",
                                   header=not os.path.exists(out_csv), index=False)
        msg = (f"[{args.tag or 'run'}] seed={seed} ep={best_epoch} "
               f"val mcc={best_val['mcc']:+.4f} acc={best_val['acc']:.4f} "
               f"wd={best_val['within_day_mcc']:+.4f} "
               f"collapse={best_val['collapse_mcc']:+.4f} ({row['minutes']}m)")
        if args.eval_test:
            msg += (f" || TEST mcc={tm['mcc']:+.4f} acc={tm['acc']:.4f} "
                    f"wd={tm['within_day_mcc']:+.4f} collapse={tm['collapse_mcc']:+.4f}")
        print(msg, flush=True)


if __name__ == "__main__":
    main()
