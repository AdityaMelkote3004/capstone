"""Experiment 13 -- save day-level summaries of a trained model's hidden states
for the market-information probe (see PREREGISTRATION.md, Experiment 13).

Called once after training, selection and test evaluation, in eval mode and
without gradients, so it consumes no random numbers and cannot change any
reported score.

For every day with at least two window-valid stocks and for every representation
R (n_valid x h) the file stores [mean_i R, std_i R] over WINDOW-VALID stocks
(labelled stocks would depend on the target day's move). Representations:

  MMGT (DTML)        "mmgt"            final stock states (out_norm output)
  MMGT-D2            "timing"          the timing branch's final stock states
                     "select"          the selection branch's final states
                     "select_centred"  the same, centred over valid stocks: what
                                       the bias-free excess head actually sees
                                       (its mean is zero by construction, so only
                                       the std half carries information)

For test days it also stores per-stock arrays (window-valid stocks only) of the
final logit, the timing-branch logit l and the excess logit e where they exist,
and the realised return, for Rank IC of each branch.
"""

import os

import numpy as np
import torch

from src.models.dtml_disentangled import DTMLTwoBranch


def _hook_targets(model):
    if isinstance(model, DTMLTwoBranch):
        return {"timing": model.timing.out_norm, "select": model.select.out_norm}
    return {"mmgt": model.out_norm}


@torch.no_grad()
def collect_states(model, data, extra, split_idx):
    model.eval()
    captured, hooks = {}, []
    for name, mod in _hook_targets(model).items():
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, name=name: captured.__setitem__(name, o.detach())))
    summ, day_ids, splits = {}, [], []
    per_stock = {k: [] for k in ("day", "stock", "logit", "l", "e", "ret")}
    try:
        for split, idx in split_idx.items():
            for i in idx:
                valid = torch.from_numpy(data["window_ok"][i])
                if valid.sum() < 2:
                    continue
                logits = model(torch.from_numpy(data["x"][i]), valid,
                               torch.from_numpy(data["market"][i]),
                               None if extra is None else torch.from_numpy(extra[i]))
                reps = {k: v[valid] for k, v in captured.items()}
                if "select" in reps:
                    reps["select_centred"] = reps["select"] - reps["select"].mean(0, keepdim=True)
                for k, r in reps.items():
                    summ.setdefault(k, []).append(torch.cat([r.mean(0), r.std(0)]).numpy())
                day_ids.append(i)
                splits.append(split)
                if split == "test":
                    keep = valid.numpy()
                    s = np.where(keep)[0]
                    per_stock["day"].append(np.full(len(s), i))
                    per_stock["stock"].append(s)
                    per_stock["logit"].append(logits.numpy()[keep])
                    per_stock["ret"].append(data["ret"][i][keep])
                    parts = getattr(model, "parts", {})
                    if "timing_logits" in parts:
                        per_stock["l"].append(parts["timing_logits"].numpy()[keep])
                        per_stock["e"].append(parts["e"].numpy()[keep])
    finally:
        for h in hooks:
            h.remove()
    out = {f"rep_{k}": np.stack(v) for k, v in summ.items()}
    out.update({"day_idx": np.array(day_ids), "split": np.array(splits)})
    out.update({f"test_{k}": np.concatenate(v) for k, v in per_stock.items() if v})
    return out


def save_states(path, states):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **states)
