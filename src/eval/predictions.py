"""Save every test prediction so date-level statistics and the backtest can be
computed after training.

Predictions are kept for ALL stocks with a complete input window on each day,
labelled or not. The backtest must rank this full set: restricting it to labelled
stocks would use target-day information (labelled = moved outside the buffer
zone), the same leak that was removed from the stock profile.
"""

import os

import numpy as np
import torch


@torch.no_grad()
def collect(model, data, idx, extra=None, adj=None):
    model.eval()
    rows = {"day": [], "stock": [], "prob": [], "label": [], "ret": []}
    for i in idx:
        valid = torch.from_numpy(data["window_ok"][i])
        if valid.sum() < 2:
            continue
        logits = model(torch.from_numpy(data["x"][i]), valid,
                       torch.from_numpy(data["market"][i]),
                       None if extra is None else torch.from_numpy(extra[i]),
                       None if adj is None else torch.from_numpy(adj[i]))
        keep = valid.numpy()
        s = np.where(keep)[0]
        rows["day"].append(np.full(len(s), i))
        rows["stock"].append(s)
        rows["prob"].append(torch.sigmoid(logits).numpy()[keep])
        rows["label"].append(data["y"][i][keep])
        rows["ret"].append(data["ret"][i][keep])
        # disentangled models also expose their market / excess / beta components
        for k in ("m", "e", "b"):
            part = getattr(model, "parts", {}).get(k)
            if part is not None:
                rows.setdefault(k, []).append(part.numpy()[keep])
    return {k: np.concatenate(v) for k, v in rows.items() if v}


def save(path, preds, dates):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, dates=np.array(dates), **preds)
