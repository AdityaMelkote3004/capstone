import torch

from src.models.graph_model import StockGAT, build_sector_graph, edge_index_for_tickers


def _write_sector_csv(tmp_path):
    path = tmp_path / "sectors.csv"
    path.write_text("Ticker,Sector\nAAA,Tech\nBBB,Tech\nCCC,Energy\nDDD,Energy\n")
    return str(path)


def test_build_sector_graph_connects_same_sector_only(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    assert graph.number_of_nodes() == 4
    assert graph.has_edge("AAA", "BBB")
    assert graph.has_edge("CCC", "DDD")
    assert not graph.has_edge("AAA", "CCC")
    assert graph.nodes["AAA"]["sector"] == "Tech"


def test_edge_index_for_tickers_uses_local_indices(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    edge_index = edge_index_for_tickers(graph, ["AAA", "BBB", "CCC"])
    edge_set = set(map(tuple, edge_index.t().tolist()))
    assert (0, 1) in edge_set and (1, 0) in edge_set
    assert (0, 2) not in edge_set and (1, 2) not in edge_set


def test_edge_index_for_tickers_empty_when_no_shared_sector(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    edge_index = edge_index_for_tickers(graph, ["AAA", "CCC"])
    assert edge_index.shape == (2, 0)


def test_stock_gat_forward_shape():
    model = StockGAT(price_input_dim=3, hidden_dim=16, heads=2)
    price_seq = torch.randn(4, 4, 3)
    price_mask = torch.ones(4, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    logits = model(price_seq, price_mask, edge_index)
    assert logits.shape == (4, 2)


def test_stock_gat_forward_with_no_edges():
    model = StockGAT(price_input_dim=3, hidden_dim=16, heads=2)
    price_seq = torch.randn(3, 4, 3)
    price_mask = torch.ones(3, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    logits = model(price_seq, price_mask, edge_index)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
