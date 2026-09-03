import pickle

from src.data.edge_cache import build_edge_cache, load_edge_cache


def test_builds_and_roundtrips_a_cache(tmp_path):
    out = tmp_path / "edges.pkl"
    cache = build_edge_cache(
        "dataset/stocknet_protocol_a_exact.parquet",
        "dataset/final/stocknet_final_modeling_set.parquet",
        "dataset/final/sector_mapping.csv",
        str(out),
    )
    assert out.exists()
    assert len(cache) == 504, "one entry per trading day in the exact-protocol set"

    reloaded = load_edge_cache(str(out))
    assert reloaded.keys() == cache.keys()

    sample = cache[sorted(cache)[300]]
    assert set(sample) == {"sector", "correlation", "comention"}
    assert len(sample["sector"]) == 381
    assert len(sample["correlation"]) > 0


def test_early_dates_have_empty_correlation_but_full_sector(tmp_path):
    """Warm-up behaviour: the first trading days lack the history for a
    60-day correlation window, and must degrade to empty, not crash."""
    cache = build_edge_cache(
        "dataset/stocknet_protocol_a_exact.parquet",
        "dataset/final/stocknet_final_modeling_set.parquet",
        "dataset/final/sector_mapping.csv",
        str(tmp_path / "edges.pkl"),
    )
    first = cache[sorted(cache)[0]]
    assert first["correlation"] == set()
    assert len(first["sector"]) == 381
