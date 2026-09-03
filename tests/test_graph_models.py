from itertools import combinations

import torch

from src.models.graph_models import StockGraphSAGE, StockRGCN


def _clique(n):
    src, dst = [], []
    for a, b in combinations(range(n), 2):
        src.extend([a, b])
        dst.extend([b, a])
    return torch.tensor([src, dst], dtype=torch.long)


def test_graphsage_forward_shape():
    model = StockGraphSAGE(hidden_dim=16)
    logits = model(torch.randn(5, 4, 3), torch.ones(5, 4), _clique(5))
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_rgcn_forward_shape():
    model = StockRGCN(hidden_dim=16)
    ei = _clique(5)
    et = torch.zeros(ei.shape[1], dtype=torch.long)
    logits = model(torch.randn(5, 4, 3), torch.ones(5, 4), ei, et)
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_rgcn_handles_all_three_relations():
    model = StockRGCN(hidden_dim=16, num_relations=3)
    ei = _clique(6)
    et = torch.randint(0, 3, (ei.shape[1],))
    logits = model(torch.randn(6, 4, 3), torch.ones(6, 4), ei, et)
    assert logits.shape == (6, 2)
    assert torch.isfinite(logits).all()


def test_models_handle_empty_edge_index():
    """Early dates can have no edges at all; models must still run."""
    ei = torch.empty((2, 0), dtype=torch.long)
    et = torch.empty((0,), dtype=torch.long)
    for logits in (
        StockGraphSAGE(hidden_dim=16)(torch.randn(3, 4, 3), torch.ones(3, 4), ei),
        StockRGCN(hidden_dim=16)(torch.randn(3, 4, 3), torch.ones(3, 4), ei, et),
    ):
        assert logits.shape == (3, 2)
        assert torch.isfinite(logits).all()


def test_graphsage_does_not_collapse_on_a_clique_after_training():
    """Experiment 4's over-smoothing regression, ported: on a complete
    clique, node outputs must stay distinguishable after training."""
    torch.manual_seed(0)
    n = 8
    model = StockGraphSAGE(hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, mask, ei = torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(x, mask, ei), target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x, mask, ei)
    assert logits.std(dim=0).mean().item() > 1e-4


def test_rgcn_does_not_collapse_on_a_clique_after_training():
    torch.manual_seed(0)
    n = 8
    model = StockRGCN(hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, mask, ei = torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n)
    et = torch.zeros(ei.shape[1], dtype=torch.long)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(x, mask, ei, et), target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x, mask, ei, et)
    assert logits.std(dim=0).mean().item() > 1e-4


def test_rgcn_synthesizes_edge_type_when_omitted():
    """The brief requires StockRGCN to synthesize a zero edge_type when the
    caller omits it, so every model can be called with one signature."""
    model = StockRGCN(hidden_dim=16)
    ei = _clique(5)
    logits = model(torch.randn(5, 4, 3), torch.ones(5, 4), ei)  # no edge_type
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_rgcn_synthesizes_edge_type_when_omitted_with_empty_edges():
    """Edge case: empty edge_index with omitted edge_type still synthesizes correctly."""
    model = StockRGCN(hidden_dim=16)
    ei = torch.empty((2, 0), dtype=torch.long)
    logits = model(torch.randn(3, 4, 3), torch.ones(3, 4), ei)  # no edge_type
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
