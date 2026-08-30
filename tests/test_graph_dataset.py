import pandas as pd
import torch

from src.data.graph_dataset import build_daily_snapshots
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
