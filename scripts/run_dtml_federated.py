"""Experiment 11 stage F -- the F1 model trained with federated averaging.

Clients are the nine StockNet sectors (7-10 tickers each). Each round, every
client starts from the global weights and trains locally for `local_epochs`
over the training days, seeing only its own stocks: the attention mask and the
loss are restricted to the client's tickers, so no client's stocks enter another
client's forward pass. The server averages client weights in proportion to the
number of labelled training samples each client holds (FedAvg). The global
model is selected on validation MCC over rounds and evaluated on all stocks.

Shared, public inputs: the equal-weighted market index and the graph-derived
peer features use public price data of other stocks; in this simulation all
inputs are public market data, so federation here measures the OPTIMISATION
cost of partitioned training, not a privacy guarantee (no secure aggregation or
differential privacy is applied).

Config, features and selection rule are identical to F1 so that the comparison
isolates federation.
"""

import argparse
import copy
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
from scripts.run_dtml_plus import run_split
from src.data.acl18 import load_acl18
from src.data.acl18_extra import extra_features
from src.models.dtml import DTML
from src.utils.seed import set_seed

OUT_DIR = "results/experiment11_dtml"
SECTOR_CSV = "dataset/final/sector_mapping.csv"


def local_train(model, data, extra, days, client_mask, lr, reg, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    cm = torch.from_numpy(client_mask)
    for _ in range(epochs):
        order = list(days)
        random.shuffle(order)
        for i in order:
            valid = torch.from_numpy(data["window_ok"][i]) & cm
            y = torch.from_numpy(data["y"][i])
            lab = (y >= 0) & cm
            if valid.sum() < 2 or not lab.any():
                continue
            logits = model(torch.from_numpy(data["x"][i]), valid,
                           torch.from_numpy(data["market"][i]),
                           torch.from_numpy(extra[i]))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[lab], y[lab].float()) + reg * model.selective_penalty()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="FED_fedavg")
    args = ap.parse_args()

    data = load_acl18(window=10)
    extra = extra_features(data, peer=True)
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}
    sectors = pd.read_csv(SECTOR_CSV).set_index("Ticker")["Sector"]
    names = sorted(set(sectors))
    masks = {s: np.array([sectors[t] == s for t in data["tickers"]]) for s in names}
    weights = {s: sum(int(((data["y"][i] >= 0) & masks[s]).sum()) for i in idx["train"])
               for s in names}
    total = sum(weights.values())

    for seed in args.seeds:
        set_seed(seed)
        t0 = time.time()
        glob = DTML(n_stocks=len(data["tickers"]), hidden=64, heads=4, beta=1.0,
                    dropout=0.15, extra_dim=extra.shape[-1])
        best, best_state, best_round, stale = -9.0, None, 0, 0
        for rnd in range(args.rounds):
            avg = None
            for s in names:
                local = copy.deepcopy(glob)
                sd = local_train(local, data, extra, idx["train"], masks[s],
                                 1e-4, 1.0, args.local_epochs)
                w = weights[s] / total
                avg = ({k: v.detach().clone() * w for k, v in sd.items()} if avg is None
                       else {k: avg[k] + sd[k].detach() * w for k in avg})
            glob.load_state_dict(avg)
            vm = metrics(*run_split(glob, data, extra, idx["val"]))
            if vm["mcc"] > best:
                best, best_round, stale, best_val = vm["mcc"], rnd, 0, vm
                best_state = {k: v.clone() for k, v in glob.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
        glob.load_state_dict(best_state)
        tm = metrics(*run_split(glob, data, extra, idx["test"]), boot=2000, seed=seed)
        row = {"tag": args.tag, "seed": seed, "rounds_run": rnd + 1,
               "best_round": best_round, "local_epochs": args.local_epochs,
               "clients": len(names), "minutes": round((time.time() - t0) / 60, 1)}
        row.update({f"val_{k}": v for k, v in best_val.items()})
        row.update({f"test_{k}": v for k, v in tm.items()})
        out = os.path.join(OUT_DIR, "federated_final.csv")
        pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"[{args.tag}] seed={seed} best_round={best_round} val mcc={best_val['mcc']:+.4f} "
              f"|| TEST mcc={tm['mcc']:+.4f} acc={tm['acc']:.4f} auc={tm['auc']:.4f} "
              f"wd={tm['within_day_mcc']:+.4f} ({row['minutes']}m)", flush=True)


if __name__ == "__main__":
    main()
