import torch

from src.data.graph_dataset import DaySnapshot, normalize_price_snapshots
from src.data.price_fix import winsorize_price_snapshots


def _snapshot(values: list) -> DaySnapshot:
    """One snapshot with N tickers, 4 days, 3 price features."""
    seq = torch.tensor(values, dtype=torch.float32)
    n = seq.shape[0]
    return DaySnapshot(
        tickers=[f"T{i}" for i in range(n)],
        price_seq=seq,
        price_mask=torch.ones(n, 4),
        target=torch.zeros(n, dtype=torch.long),
        date="2014-01-02",
    )


def _normal_ticker() -> list:
    return [[0.1, 0.1, 0.5]] * 4


def _outlier_ticker() -> list:
    return [[0.1, 0.1, 6500.0]] * 4


def test_winsorize_clips_only_beyond_the_limit():
    snap = _snapshot([_normal_ticker(), _outlier_ticker()])
    winsorize_price_snapshots([[snap]], limit=10.0)
    assert snap.price_seq[0, 0, 2].item() == 0.5, "in-range values must be untouched"
    assert snap.price_seq[1, 0, 2].item() == 10.0, "out-of-range values clip to the limit"


def test_winsorize_clips_negative_side_too():
    snap = _snapshot([[[0.1, 0.1, -6495.0]] * 4])
    winsorize_price_snapshots([[snap]], limit=10.0)
    assert snap.price_seq[0, 0, 2].item() == -10.0


def test_winsorize_reports_what_it_clipped():
    snap = _snapshot([_normal_ticker(), _outlier_ticker()])
    report = winsorize_price_snapshots([[snap]], limit=10.0)
    assert report["clipped_per_feature"][2] == 4, "4 days of one outlier ticker"
    assert report["clipped_per_feature"][0] == 0
    assert report["total_values"] == 2 * 4 * 3


def test_winsorize_restores_dynamic_range_after_normalization():
    """THE regression: without winsorizing, one extreme ticker crushes every
    other ticker's third feature toward zero after z-scoring."""
    tickers = [_normal_ticker() for _ in range(20)]
    # give the normal tickers real spread on feature 2
    for i, t in enumerate(tickers):
        for d in range(4):
            t[d] = [0.1, 0.1, float(i) / 10.0]
    poisoned = _snapshot([*[list(t) for t in tickers], _outlier_ticker()])
    clean = _snapshot([*[list(t) for t in tickers], _outlier_ticker()])

    normalize_price_snapshots([poisoned])

    winsorize_price_snapshots([[clean]], limit=10.0)
    normalize_price_snapshots([clean])

    poisoned_spread = poisoned.price_seq[:20, :, 2].std().item()
    clean_spread = clean.price_seq[:20, :, 2].std().item()
    assert clean_spread > 10 * poisoned_spread, (
        f"winsorizing must restore spread: {clean_spread} vs {poisoned_spread}"
    )


def test_normalize_accepts_winsorize_limit_and_default_is_unchanged():
    a = _snapshot([_normal_ticker(), _outlier_ticker()])
    b = _snapshot([_normal_ticker(), _outlier_ticker()])
    normalize_price_snapshots([a])                          # default: no winsorizing
    normalize_price_snapshots([b], winsorize_limit=10.0)
    assert not torch.allclose(a.price_seq, b.price_seq)
