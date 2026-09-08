from itertools import combinations

import torch

from src.models.macro_conditioned import MacroConditionedGraphNet


def _clique(n: int) -> torch.Tensor:
    src, dst = [], []
    for a, b in combinations(range(n), 2):
        src.extend([a, b])
        dst.extend([b, a])
    return torch.tensor([src, dst], dtype=torch.long)


def _inputs(n: int = 5):
    return (torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n),
            torch.randn(n, 1536), torch.randn(771))


def test_forward_shape():
    model = MacroConditionedGraphNet(hidden_dim=16)
    logits = model(*_inputs(5))
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_empty_edge_index_still_runs():
    model = MacroConditionedGraphNet(hidden_dim=16)
    price, mask, _, micro, macro = _inputs(3)
    logits = model(price, mask, torch.empty((2, 0), dtype=torch.long), micro, macro)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_macro_only_gives_every_ticker_the_same_prediction():
    """Row 0's whole purpose: it CANNOT discriminate between stocks."""
    model = MacroConditionedGraphNet(hidden_dim=16, macro_only=True)
    model.eval()
    price, mask, edges, micro, macro = _inputs(6)
    with torch.no_grad():
        logits = model(price, mask, edges, micro, macro)
    assert logits.shape == (6, 2)
    assert torch.allclose(logits, logits[0].expand_as(logits), atol=1e-6)


def test_conditioning_none_ignores_macro():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="none")
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, micro, torch.randn(771))
    assert torch.allclose(a, b), "conditioning='none' must not read macro"


def test_micro_conditioning_changes_the_output():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="micro")
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, micro, torch.randn(771) * 5)
    assert not torch.allclose(a, b)


def test_film_parameters_are_exposed_for_the_explainability_readout():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="micro")
    model(*_inputs(4))
    gamma, beta = model.last_film
    assert gamma.shape == (model.micro_proj,)
    assert beta.shape == (model.micro_proj,)


def test_use_micro_false_ignores_micro():
    model = MacroConditionedGraphNet(hidden_dim=16, use_micro=False)
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, torch.randn(4, 1536), macro)
    assert torch.allclose(a, b)


def test_use_graph_false_ignores_edges():
    model = MacroConditionedGraphNet(hidden_dim=16, use_graph=False)
    model.eval()
    price, mask, _, micro, macro = _inputs(5)
    with torch.no_grad():
        a = model(price, mask, _clique(5), micro, macro)
        b = model(price, mask, torch.empty((2, 0), dtype=torch.long), micro, macro)
    assert torch.allclose(a, b)


def test_attention_conditioning_reads_macro():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="attention")
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, micro, torch.randn(771) * 5)
    assert not torch.allclose(a, b)


def test_does_not_collapse_on_a_clique_after_training():
    """Experiment 4's over-smoothing regression, ported."""
    torch.manual_seed(0)
    n = 8
    model = MacroConditionedGraphNet(hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    price, mask, edges = torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n)
    micro, macro = torch.randn(n, 1536), torch.randn(771)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(
            model(price, mask, edges, micro, macro), target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(price, mask, edges, micro, macro)
    assert logits.std(dim=0).mean().item() > 1e-4
