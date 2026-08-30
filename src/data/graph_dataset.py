"""Builds per-trading-day graph snapshots for Experiment 4's sector-graph
GAT: each snapshot is the cross-section of tickers with a valid sample on
one Target_Date, the unit the GAT attends over. Reuses
dataset/stocknet_protocol_a_exact.parquet's existing chronological
train/dev/test split -- no new splitting logic here.
"""

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass
class DaySnapshot:
    tickers: list
    price_seq: torch.Tensor   # (N, 4, 3)
    price_mask: torch.Tensor  # (N, 4)
    target: torch.Tensor      # (N,)


def _row_to_price_seq(row: pd.Series) -> torch.Tensor:
    seq = torch.zeros(4, 3)
    for d in range(4):
        for p in range(3):
            seq[d, p] = row[f"Day{d + 1}_Price{p + 1}"]
    return seq


def _window_mask(window_length: int) -> torch.Tensor:
    mask = torch.zeros(4)
    mask[: min(int(window_length), 4)] = 1.0
    return mask


def normalize_price_snapshots(train_snapshots: list, *other_snapshot_lists: list) -> None:
    """Z-scores price_seq in place across a list of DaySnapshot lists, using
    mean/std computed only from `train_snapshots` (flattened across all
    nodes and days per one of the 3 price features), matching Experiment 3's
    `normalize_price` (preprocessing_notebooks/MMGTFFF_mansf_joint_bilinear.ipynb,
    cell 16): train statistics only, applied to train and every list in
    `other_snapshot_lists` (e.g. dev, test)."""
    flat = torch.cat([snap.price_seq.reshape(-1, 3) for snap in train_snapshots], dim=0)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)

    for snap_list in (train_snapshots, *other_snapshot_lists):
        for snap in snap_list:
            snap.price_seq = (snap.price_seq - mean) / std


def build_daily_snapshots(df: pd.DataFrame, split: str) -> list:
    """Groups rows of `df` where Split == split by Target_Date, sorted
    chronologically, returning one DaySnapshot per date."""
    subset = df[df["Split"] == split].sort_values("Target_Date")
    snapshots = []
    for _, day_df in subset.groupby("Target_Date", sort=True):
        tickers = day_df["Ticker"].tolist()
        price_seq = torch.stack([_row_to_price_seq(row) for _, row in day_df.iterrows()])
        price_mask = torch.stack([_window_mask(wl) for wl in day_df["Window_Length"]])
        target = torch.tensor(day_df["Target"].values, dtype=torch.long)
        snapshots.append(DaySnapshot(tickers=tickers, price_seq=price_seq,
                                      price_mask=price_mask, target=target))
    return snapshots
