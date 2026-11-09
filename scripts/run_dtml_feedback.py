"""Experiment 11 stage B -- DTML with outcome feedback (online fine-tuning).

The "feedback" in feedback-based improvement, made concrete and leak-free:

  for each evaluation day t, in date order:
      1. predict day t with the current weights            (scored)
      2. day t closes; its labels become known
      3. take `steps` gradient steps on day t's labels       (feedback)
  day t+1 is then predicted by weights that have seen outcomes up to t only.

Motivation: train ends 2015-07-31 and test is Q4 2015, so the model is scored on
a regime it never saw. Incremental fine-tuning on realised outcomes is the
standard remedy for that drift (e.g. DoubleAdapt, KDD 2023), and it costs one
small backward pass per day.

Leakage argument: step 3 for day t runs strictly after step 1 for day t, and no
day's prediction ever uses its own label or any later one. For test, the model
first streams through the validation days (whose labels are all known before
2015-10-01), then through test.

Risk this script is built to expose: fine-tuning on yesterday's labels drags
every stock toward yesterday's market direction. That is market timing, not
stock picking, and the collapse control and within-day MCC report it directly.

The base model is trained exactly as in run_dtml.py and selected on offline
validation; the feedback hyperparameters (steps, lr) are chosen on validation
too. Test stays locked unless --eval-test.
"""

import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.run_dtml import metrics, run_split
from src.data.acl18 import load_acl18
from src.models.dtml import DTML
from src.utils.seed import set_seed

OUT_DIR = "results/experiment11_dtml"


def feedback_stream(model, data, warm_idx, eval_idx, steps, lr, reg):
    """Predict-then-update over eval_idx, after silently updating on warm_idx."""
    model = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def update(i):
        valid = torch.from_numpy(data["window_ok"][i])
        y = torch.from_numpy(data["y"][i])
        labelled = y >= 0
        if valid.sum() < 2 or not labelled.any():
            return
        model.train()
        x = torch.from_numpy(data["x"][i])
        m = torch.from_numpy(data["market"][i])
        for _ in range(steps):
            logits = model(x, valid, m)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[labelled], y[labelled].float()) + reg * model.selective_penalty()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    for i in sorted(warm_idx):
        update(i)
    probs, ys, days = [], [], []
    for i in sorted(eval_idx):
        p, y, d = run_split(model, data, [i])   # eval mode, no update
        if len(y):
            probs.append(p); ys.append(y); days.append(d)
        update(i)                                # outcome now known
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(days)


def train_base(args, data, idx, seed):
    set_seed(seed)
    model = DTML(n_stocks=len(data["tickers"]), hidden=args.hidden, heads=args.heads,
                 beta=args.beta, dropout=args.dropout,
                 use_mc=args.ablate != "mc", use_sa=args.ablate != "sa")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, state, stale = -9.0, None, 0
    for _ in range(args.epochs):
        run_split(model, data, idx["train"], opt, args.reg)
        vm = metrics(*run_split(model, data, idx["val"]))
        if vm["mcc"] > best:
            best, stale = vm["mcc"], 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(state)
    return model


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
    ap.add_argument("--ablate", default="none", choices=["none", "sa", "mc"])
    ap.add_argument("--fb-steps", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--fb-lr", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="fb")
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()

    data = load_acl18(window=args.window)
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}
    out_csv = os.path.join(OUT_DIR, "feedback_final.csv" if args.eval_test
                           else "feedback_search.csv")

    for seed in args.seeds:
        t0 = time.time()
        base = train_base(args, data, idx, seed)
        settings = [(0, 0.0)] + [(s, l) for s in args.fb_steps for l in args.fb_lr]
        for steps, fb_lr in settings:
            if steps == 0:
                vm = metrics(*run_split(base, data, idx["val"]))
            else:
                vm = metrics(*feedback_stream(base, data, [], idx["val"],
                                              steps, fb_lr, args.reg))
            row = {"tag": args.tag, "seed": seed, "window": args.window,
                   "beta": args.beta, "hidden": args.hidden, "lr": args.lr,
                   "ablate": args.ablate, "fb_steps": steps, "fb_lr": fb_lr}
            row.update({f"val_{k}": v for k, v in vm.items()})
            msg = (f"[{args.tag}] seed={seed} fb={steps}x{fb_lr:g} val "
                   f"mcc={vm['mcc']:+.4f} wd={vm['within_day_mcc']:+.4f} "
                   f"collapse={vm['collapse_mcc']:+.4f}")
            if args.eval_test:
                if steps == 0:
                    tm = metrics(*run_split(base, data, idx["test"]), boot=2000, seed=seed)
                else:
                    tm = metrics(*feedback_stream(base, data, idx["val"], idx["test"],
                                                  steps, fb_lr, args.reg),
                                 boot=2000, seed=seed)
                row.update({f"test_{k}": v for k, v in tm.items()})
                msg += (f" || TEST mcc={tm['mcc']:+.4f} acc={tm['acc']:.4f} "
                        f"wd={tm['within_day_mcc']:+.4f} collapse={tm['collapse_mcc']:+.4f}")
            pd.DataFrame([row]).to_csv(out_csv, mode="a",
                                       header=not os.path.exists(out_csv), index=False)
            print(msg, flush=True)
        print(f"  seed {seed} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
