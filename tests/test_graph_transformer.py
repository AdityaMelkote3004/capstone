from itertools import combinations

import torch

from src.models.graph_model import build_sector_graph
from src.models.graph_transformer import StockGraphTransformer, build_sector_bias


def _write_sector_csv(tmp_path):
    path = tmp_path / "sectors.csv"
    path.write_text("Ticker,Sector\nAAA,Tech\nBBB,Tech\nCCC,Energy\nDDD,Energy\n")
    return str(path)


def test_build_sector_bias_marks_same_sector_pairs_only(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    mask = build_sector_bias(["AAA", "BBB", "CCC"], graph)
    assert mask.shape == (3, 3)
    assert mask[0, 1].item() == 1.0 and mask[1, 0].item() == 1.0  # AAA, BBB: same sector
    assert mask[0, 2].item() == 0.0 and mask[1, 2].item() == 0.0  # different sector
    assert mask[0, 0].item() == 0.0  # diagonal always zero


def test_build_sector_bias_symmetric(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    mask = build_sector_bias(["AAA", "BBB", "CCC", "DDD"], graph)
    assert torch.equal(mask, mask.t())


def test_build_sector_bias_zero_when_no_shared_sector(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    mask = build_sector_bias(["AAA", "CCC"], graph)
    assert torch.equal(mask, torch.zeros(2, 2))


def test_stock_graph_transformer_forward_shape():
    model = StockGraphTransformer(price_input_dim=3, hidden_dim=16, nhead=2, dim_feedforward=32)
    price_seq = torch.randn(5, 4, 3)
    price_mask = torch.ones(5, 4)
    same_sector_mask = torch.zeros(5, 5)
    logits = model(price_seq, price_mask, same_sector_mask)
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_stock_graph_transformer_bias_changes_output():
    """The learned sector-bias parameter must actually influence attention:
    a non-zero same_sector_mask must produce different output than an
    all-zero one, given the same inputs and weights."""
    torch.manual_seed(0)
    model = StockGraphTransformer(price_input_dim=3, hidden_dim=16, nhead=2, dim_feedforward=32)
    model.eval()
    # Force the learned bias scalar away from its near-zero init so the
    # test isn't relying on a lucky random initialization.
    with torch.no_grad():
        model.sector_bias.fill_(5.0)
    price_seq = torch.randn(5, 4, 3)
    price_mask = torch.ones(5, 4)
    zero_mask = torch.zeros(5, 5)
    biased_mask = torch.zeros(5, 5)
    biased_mask[0, 1] = biased_mask[1, 0] = 1.0
    with torch.no_grad():
        out_zero = model(price_seq, price_mask, zero_mask)
        out_biased = model(price_seq, price_mask, biased_mask)
    assert not torch.allclose(out_zero, out_biased)


def test_stock_graph_transformer_does_not_collapse_after_training():
    """Global attention can still over-smooth with enough layers/training,
    just less severely than GAT's local message passing over a complete
    graph (see Experiment 4's diagnosed bug). This regression test trains
    briefly on a synthetic all-same-sector clique and asserts node outputs
    don't collapse to a shared constant."""
    torch.manual_seed(0)
    n = 8
    model = StockGraphTransformer(price_input_dim=3, hidden_dim=16, nhead=2, dim_feedforward=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    price_seq = torch.randn(n, 4, 3)
    price_mask = torch.ones(n, 4)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    same_sector_mask = torch.ones(n, n) - torch.eye(n)

    for _ in range(30):
        logits = model(price_seq, price_mask, same_sector_mask)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(price_seq, price_mask, same_sector_mask)
    assert logits.std(dim=0).mean().item() > 1e-4
    # Nonzero logit variance alone isn't enough: the real Experiment 5 run
    # collapsed to predicting one class ~99.95% of the time while presumably
    # still having nonzero logit variance, so also assert the model actually
    # uses both classes on this balanced synthetic target.
    assert len(torch.unique(logits.argmax(dim=-1))) == 2
