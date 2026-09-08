"""Winsorization for the protocol-A price features.

`Price3` (StockNet's third precomputed price ratio, `data[5]` in
price/preprocessed/{TICKER}.txt) is price-scale-sensitive: across the 87-ticker
universe it ranges [-6495, +6500] with std 154.0, but excluding BRK-A and PCLN
-- the two highest-priced stocks, 255 rows of 26,619 -- it ranges [-44.3, +93.1]
with std 1.41.

Because `normalize_price_snapshots` computes one global mean/std across all
tickers, those two tickers inflate std 109x and compress every other ticker's
post-z-score range from +/-3.53 to +/-0.103. A third of the node features is
effectively constant zero as a result.

Winsorizing rather than dropping the tickers: BRK-A is a legitimate universe
member, and dropping it would move test n_samples off 3720 and break
comparability with Experiments 1/3/4/5/6.
"""

import torch


def winsorize_price_snapshots(snapshot_lists: list, limit: float = 10.0) -> dict:
    """Clips every price feature into [-limit, +limit], in place, across all
    DaySnapshot lists given. Returns a report of how many values were clipped
    per feature index, so the write-up can state exactly what was altered.

    Applied BEFORE normalization, and using a fixed absolute limit rather than a
    data-derived percentile, so the operation does not depend on which split it
    sees and cannot leak test-set information into training.
    """
    clipped = [0, 0, 0]
    total = 0
    for snap_list in snapshot_lists:
        for snap in snap_list:
            seq = snap.price_seq
            total += seq.numel()
            for feature in range(seq.shape[-1]):
                channel = seq[..., feature]
                clipped[feature] += int((channel.abs() > limit).sum().item())
            snap.price_seq = seq.clamp(min=-limit, max=limit)
    return {"clipped_per_feature": clipped, "total_values": total, "limit": limit}
