from itertools import combinations

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


def test_stock_gat_does_not_collapse_nodes_on_a_complete_clique_after_training():
    """Regression test for a diagnosed over-smoothing bug: on the sector
    graph (a set of disjoint complete cliques -- every ticker connected to
    every other ticker in its sector), two plain (non-residual) GATConv
    layers were empirically found to collapse every node's output to a
    bit-identical constant regardless of its own distinct input, after a
    few epochs of training, independent of hyperparameters (an untrained,
    freshly-initialized model does not show the collapse -- it only
    emerges once gradient descent has run, so this test trains briefly
    before checking, mirroring how the bug was actually found). The
    residual connections in StockGAT.forward (h = h + GATConv(h)) exist
    specifically to prevent this. This test fails if that regression
    reappears."""
    torch.manual_seed(0)
    n = 8
    model = StockGAT(price_input_dim=3, hidden_dim=16, heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    price_seq = torch.randn(n, 4, 3)
    price_mask = torch.ones(n, 4)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    src, dst = [], []
    for a, b in combinations(range(n), 2):
        src.extend([a, b])
        dst.extend([b, a])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    for _ in range(30):
        logits = model(price_seq, price_mask, edge_index)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(price_seq, price_mask, edge_index)
    # Different tickers with genuinely different price histories must not
    # produce bit-identical logits after training on a complete-clique
    # graph -- that exact symptom (std == 0.0 across nodes) is what over-
    # smoothing produced before the residual fix.
    assert logits.std(dim=0).mean().item() > 1e-4
