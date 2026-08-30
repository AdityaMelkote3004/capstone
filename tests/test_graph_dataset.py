import pandas as pd
import torch

from src.data.graph_dataset import DaySnapshot, build_daily_snapshots, normalize_price_snapshots
from src.models.graph_model import build_sector_graph, edge_index_for_tickers


def _make_df():
    rows = []
    for ticker, date, split, window_length, target in [
        ("AAA", "2020-01-01", "train", 2, 1),
        ("BBB", "2020-01-01", "train", 2, 0),
        ("AAA", "2020-01-02", "train", 3, 0),
        ("CCC", "2020-01-02", "train", 1, 1),
        ("BBB", "2020-01-03", "dev", 4, 1),
    ]:
        row = {
            "Ticker": ticker, "Target_Date": date, "Split": split,
            "Window_Length": window_length, "Target": target,
        }
        for d in range(1, 5):
            for p in range(1, 4):
                row[f"Day{d}_Price{p}"] = 0.1 * d + 0.01 * p
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_daily_snapshots_groups_by_date_within_split():
    snapshots = build_daily_snapshots(_make_df(), "train")
    assert len(snapshots) == 2
    assert snapshots[0].tickers == ["AAA", "BBB"]
    assert snapshots[1].tickers == ["AAA", "CCC"]
    assert snapshots[0].price_seq.shape == (2, 4, 3)
    assert snapshots[0].target.tolist() == [1, 0]


def test_build_daily_snapshots_respects_window_length_mask():
    snapshots = build_daily_snapshots(_make_df(), "train")
    # AAA on 2020-01-01 has Window_Length=2 -> first two days real, rest padding.
    assert snapshots[0].price_mask[0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_build_daily_snapshots_filters_to_requested_split():
    snapshots = build_daily_snapshots(_make_df(), "dev")
    assert len(snapshots) == 1
    assert snapshots[0].tickers == ["BBB"]


def test_snapshot_edge_index_matches_sector_graph(tmp_path):
    sector_csv = tmp_path / "sectors.csv"
    sector_csv.write_text("Ticker,Sector\nAAA,Tech\nBBB,Tech\nCCC,Energy\n")
    graph = build_sector_graph(str(sector_csv))
    snapshots = build_daily_snapshots(_make_df(), "train")
    day1_edges = edge_index_for_tickers(graph, snapshots[0].tickers)
    day2_edges = edge_index_for_tickers(graph, snapshots[1].tickers)
    assert day1_edges.shape[1] == 2  # AAA<->BBB, both directions
    assert day2_edges.shape[1] == 0  # AAA (Tech) and CCC (Energy) don't connect


def _make_snapshot(values_per_node, mask_value=1.0):
    """Builds a minimal DaySnapshot with given per-node (4, 3) price_seq values."""
    price_seq = torch.tensor(values_per_node, dtype=torch.float32)
    n = price_seq.shape[0]
    price_mask = torch.full((n, 4), mask_value)
    target = torch.zeros(n, dtype=torch.long)
    tickers = [f"T{i}" for i in range(n)]
    return DaySnapshot(tickers=tickers, price_seq=price_seq, price_mask=price_mask, target=target)


def test_normalize_price_snapshots_train_has_zero_mean_unit_std():
    # Two snapshots, two nodes each, 4 days x 3 features. Feature 0 has a huge
    # outlier scale like real Price3 column; feature 1/2 are small/constant.
    snap1 = _make_snapshot([
        [[1000.0, 5.0, 2.0]] * 4,
        [[-1000.0, -5.0, 2.0]] * 4,
    ])
    snap2 = _make_snapshot([
        [[500.0, 1.0, 2.0]] * 4,
        [[-500.0, -1.0, 2.0]] * 4,
    ])
    train_snaps = [snap1, snap2]
    normalize_price_snapshots(train_snaps)
    flat = torch.cat([s.price_seq.reshape(-1, 3) for s in train_snaps], dim=0)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)
    assert torch.allclose(mean, torch.zeros(3), atol=1e-5)
    # feature 0 and 1 have real spread -> std ~1; feature 2 constant -> handled separately
    assert torch.allclose(std[:2], torch.ones(2), atol=1e-4)


def test_normalize_price_snapshots_applies_train_stats_to_other_splits():
    train_snap = _make_snapshot([
        [[10.0, 0.0, 0.0]] * 4,
        [[-10.0, 0.0, 0.0]] * 4,
        [[20.0, 0.0, 0.0]] * 4,
        [[-20.0, 0.0, 0.0]] * 4,
    ])
    dev_snap = _make_snapshot([
        [[10.0, 0.0, 0.0]] * 4,
    ])
    train_snaps = [train_snap]
    dev_snaps = [dev_snap]

    flat_train = train_snap.price_seq.reshape(-1, 3).clone()
    train_mean = flat_train.mean(dim=0)
    train_std = flat_train.std(dim=0, unbiased=False)
    train_std = torch.where(train_std == 0, torch.ones_like(train_std), train_std)

    normalize_price_snapshots(train_snaps, dev_snaps)

    expected_dev = (torch.tensor([10.0, 0.0, 0.0]) - train_mean) / train_std
    assert torch.allclose(dev_snap.price_seq[0, 0], expected_dev, atol=1e-5)
    # dev stats must NOT equal dev's own mean/std (dev is a single repeated
    # value, so its own std would be 0 and normalizing with it would be
    # meaningless/undefined) -- confirm train stats (non-trivial) were used.
    assert not torch.allclose(dev_snap.price_seq[0, 0], torch.zeros(3), atol=1e-5)


def test_normalize_price_snapshots_constant_feature_no_nan_or_inf():
    train_snap = _make_snapshot([
        [[1.0, 7.0, 7.0]] * 4,
        [[-1.0, 7.0, 7.0]] * 4,
    ])
    train_snaps = [train_snap]
    normalize_price_snapshots(train_snaps)
    assert torch.isfinite(train_snap.price_seq).all()
    # constant feature (std==0 guarded to 1.0) should just be mean-centered -> 0
    assert torch.allclose(train_snap.price_seq[:, :, 1:], torch.zeros(2, 4, 2), atol=1e-5)
