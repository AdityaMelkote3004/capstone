"""Experiment 10's model: multi-modal node features over a switchable graph conv.

Composes rather than forks. The price pathway is the SAME `PriceEncoder` used in
Experiments 4-6, so the price half of this model is not a new implementation, and
the four conv types are the same ones those experiments compared.

Two design choices carry the lessons of the previous nine experiments:

  `use_graph=False` runs the identical network with message passing disabled.
  Experiment 6 had no such control -- it compared graphs against other graphs,
  so when its best cell turned out to be calling market direction rather than
  ranking stocks, nothing in that experiment could have revealed it. Here the
  control shares the harness, the features, the seeds and the metrics.

  Residual connections around every conv. Experiment 4 diagnosed over-smoothing
  on a complete-clique sector graph (bit-identical node outputs after two
  layers); residuals were the fix and are kept.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, RGCNConv, SAGEConv, TransformerConv

from src.models.price_encoder import PriceEncoder

CONV_TYPES = ("gat", "rgcn", "sage", "transformer")


class MultiModalGraphNet(nn.Module):
    def __init__(self, static_dim: int, conv_type: str = "gat",
                 price_input_dim: int = 3, hidden_dim: int = 64,
                 heads: int = 4, num_relations: int = 3, layers: int = 2,
                 dropout: float = 0.3, use_graph: bool = True,
                 contrast: bool = False, cross_sectional: bool = False):
        super().__init__()
        if conv_type not in CONV_TYPES:
            raise ValueError(f"conv_type must be one of {CONV_TYPES}")
        self.conv_type = conv_type
        self.use_graph = use_graph
        # Standard message passing SMOOTHS: it averages a node toward its
        # neighbours. The within-day metric asks the opposite -- rank stocks
        # against each other -- so smoothing erases the very differences being
        # scored. `contrast` keeps the peer signal but feeds the model
        # (node - neighbour_mean), a high-pass rather than low-pass filter:
        # "how does this stock differ from its peers today".
        self.contrast = contrast
        # Each forward() call is one trading day's cross-section, so normalising
        # across nodes here removes that day's market-wide level and leaves only
        # relative position. No leakage: it uses same-day features of other
        # stocks, all known before the target.
        self.cross_sectional = cross_sectional

        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        # Fuse the two modalities into one node vector before any message
        # passing, so neighbours exchange a representation that already contains
        # price, fundamentals, counts and text rather than price alone.
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.convs = nn.ModuleList(
            self._make_conv(conv_type, hidden_dim, heads, num_relations)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.mixers = nn.ModuleList(
            nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def _make_conv(conv_type, hidden_dim, heads, num_relations):
        if conv_type == "gat":
            return GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        if conv_type == "transformer":
            return TransformerConv(hidden_dim, hidden_dim // heads, heads=heads)
        if conv_type == "sage":
            return SAGEConv(hidden_dim, hidden_dim)
        return RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                static: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor = None) -> torch.Tensor:
        if self.cross_sectional and static.shape[0] > 2:
            static = (static - static.mean(0, keepdim=True)) / (
                static.std(0, keepdim=True) + 1e-6)

        nodes = self.fuse(torch.cat(
            [self.price_encoder(price_seq, price_mask),
             self.static_encoder(static)], dim=-1))

        if self.use_graph and edge_index is not None and edge_index.numel() > 0:
            for conv, norm, mixer in zip(self.convs, self.norms, self.mixers):
                if self.conv_type == "rgcn":
                    message = conv(nodes, edge_index, edge_type)
                else:
                    message = conv(nodes, edge_index)
                if self.contrast:
                    # Peer-relative: what this node has that its neighbourhood
                    # does not. Concatenated with the node's own state so the
                    # model can use both, rather than replacing one with the other.
                    message = mixer(torch.cat([message, nodes - message], dim=-1))
                # Residual + norm: without these a 2-layer conv on a dense graph
                # collapsed every node to the same vector in Experiment 4.
                nodes = norm(nodes + self.dropout(torch.relu(message)))

        return self.head(nodes)
