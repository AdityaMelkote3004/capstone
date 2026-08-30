"""Graph transformer (global attention) model for Experiment 5.

Replaces GAT's local, neighbor-restricted attention (src/models/graph_model.py,
Experiment 4) with global self-attention: every ticker attends to every
other ticker in a day's cross-section, not just same-sector ones. Graph
structure (currently: same-sector membership) is injected as a learned
additive bias to raw attention scores, matching the mechanism used by
Graphormer-style graph transformers -- see
docs/superpowers/specs/2026-08-30-graph-transformer-design.md for why this
sidesteps the over-smoothing bug diagnosed in Experiment 4 (that bug came
from stacking *local* neighbor-averaging steps over a complete graph;
global attention has no such restriction to compound).

This is additive to Experiment 4, not a replacement -- StockGAT and its
results are untouched.
"""

import networkx as nx
import torch
import torch.nn as nn

from src.models.price_encoder import PriceEncoder


def build_sector_bias(tickers: list, graph: nx.Graph) -> torch.Tensor:
    """Builds a dense (N, N) structural mask for `tickers`: entry [i, j]
    is 1.0 if tickers[i] and tickers[j] share a sector (per `graph`,
    built by src.models.graph_model.build_sector_graph) and i != j, else
    0.0. The diagonal is always 0.0 -- self-attention is already handled
    natively by nn.TransformerEncoderLayer, no separate self bias needed.
    This mask is multiplied by a single learned scalar inside
    StockGraphTransformer to produce the actual additive attention bias."""
    n = len(tickers)
    mask = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j and graph.has_edge(tickers[i], tickers[j]):
                mask[i, j] = 1.0
    return mask


class StockGraphTransformer(nn.Module):
    """Price-only graph transformer: PriceEncoder per ticker -> 2-layer
    global self-attention (nn.TransformerEncoderLayer) with a learned
    same-sector attention bias -> per-ticker binary classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.sector_bias = nn.Parameter(torch.zeros(1))
        layer_kwargs = dict(d_model=hidden_dim, nhead=nhead,
                             dim_feedforward=dim_feedforward, dropout=dropout,
                             batch_first=True)
        self.layer1 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.layer2 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                same_sector_mask: torch.Tensor) -> torch.Tensor:
        node_features = self.price_encoder(price_seq, price_mask)  # (N, hidden_dim)
        bias = same_sector_mask * self.sector_bias  # (N, N), learned scalar at same-sector pairs
        x = node_features.unsqueeze(0)  # (1, N, hidden_dim) for batch_first Transformer layers
        x = self.layer1(x, src_mask=bias)
        x = self.layer2(x, src_mask=bias)
        h = x.squeeze(0)  # (N, hidden_dim)
        return self.classifier(h)
