import torch

from src.models.graph_transformer_multi import MultiRelationGraphTransformer


def test_forward_shape():
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    et = torch.tensor([0, 0, 1, 1])
    logits = model(torch.randn(5, 4, 3), torch.ones(5, 4), ei, et)
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_empty_edge_index_still_runs():
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    ei = torch.empty((2, 0), dtype=torch.long)
    et = torch.empty((0,), dtype=torch.long)
    logits = model(torch.randn(3, 4, 3), torch.ones(3, 4), ei, et)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_each_relation_has_its_own_learnable_bias():
    model = MultiRelationGraphTransformer(num_relations=3)
    assert model.relation_bias.shape == (3,)
    assert model.relation_bias.requires_grad


def test_relation_bias_changes_the_output():
    """A non-zero bias on a relation must actually alter attention."""
    torch.manual_seed(0)
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    model.eval()
    ei = torch.tensor([[0, 1], [1, 0]])
    et = torch.tensor([0, 0])
    x, mask = torch.randn(4, 4, 3), torch.ones(4, 4)
    with torch.no_grad():
        model.relation_bias.zero_()
        flat = model(x, mask, ei, et)
        model.relation_bias[0] = 5.0
        biased = model(x, mask, ei, et)
    assert not torch.allclose(flat, biased)


def test_relations_are_independent():
    """Changing relation 1's bias must not move a graph that only uses
    relation 0."""
    torch.manual_seed(0)
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    model.eval()
    ei = torch.tensor([[0, 1], [1, 0]])
    et = torch.tensor([0, 0])
    x, mask = torch.randn(4, 4, 3), torch.ones(4, 4)
    with torch.no_grad():
        model.relation_bias.zero_()
        before = model(x, mask, ei, et)
        model.relation_bias[1] = 5.0        # a relation not present in et
        after = model(x, mask, ei, et)
    assert torch.allclose(before, after)
