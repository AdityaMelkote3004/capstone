import torch

from src.models.price_encoder import PriceEncoder


def test_output_shape():
    encoder = PriceEncoder(input_dim=3, hidden_dim=64)
    x = torch.randn(5, 4, 3)
    mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.float32,
    )
    out = encoder(x, mask)
    assert out.shape == (5, 64)


def test_handles_single_valid_day_without_nan():
    encoder = PriceEncoder(input_dim=3, hidden_dim=64)
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    out = encoder(x, mask)
    assert torch.isfinite(out).all()
