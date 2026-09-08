"""Experiment 7's model: a daily macro event vector FiLM-conditions each
ticker's micro (tweet + news) representation before a GAT mixes tickers over a
tweet-similarity graph.

The conditioning claim is semantic: an earnings headline means something
different during a tightening cycle than during easing. Whether the attention
logits are the better conditioning site is an empirical question, tested as
`conditioning="attention"` rather than assumed.

`gamma` and `beta` are shared across every ticker on a given day. That is
deliberate -- the explainability readout plots one gamma per trading day
against the macro calendar -- but it also means macro conditioning cannot by
itself reorder tickers within a day, which is why the ablation ladder carries a
macro-only control (`macro_only=True`) and reports within-day metrics.

Residual connections wrap each GAT layer for the over-smoothing reason
documented in results/experiment4_gat/EXPERIMENT4_GAT.md.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from src.models.price_encoder import PriceEncoder


class MacroConditionedGraphNet(nn.Module):
    """One class covering every row of Experiment 7's ablation ladder."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 micro_dim: int = 1536, macro_dim: int = 771, micro_proj: int = 32,
                 heads: int = 4, dropout: float = 0.2, use_micro: bool = True,
                 use_graph: bool = True, conditioning: str = "micro",
                 macro_only: bool = False):
        super().__init__()
        if conditioning not in ("none", "micro", "attention"):
            raise ValueError(f"unknown conditioning: {conditioning}")

        self.use_micro = use_micro
        self.use_graph = use_graph
        self.conditioning = conditioning
        self.macro_only = macro_only
        self.micro_proj = micro_proj
        self.last_film = None

        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.micro_projection = nn.Linear(micro_dim, micro_proj)
        self.micro_default = nn.Parameter(torch.zeros(micro_proj))
        self.film = nn.Sequential(
            nn.Linear(macro_dim, 64), nn.ReLU(), nn.Linear(64, 2 * micro_proj),
        )
        self.attention_film = nn.Sequential(
            nn.Linear(macro_dim, 32), nn.ReLU(), nn.Linear(32, 2),
        )

        node_dim = hidden_dim + (micro_proj if use_micro else 0)
        self.gat1 = GATConv(node_dim, node_dim, heads=heads, concat=False,
                            dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(node_dim, node_dim, heads=heads, concat=False,
                            dropout=dropout, add_self_loops=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(node_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )
        self.macro_head = nn.Sequential(
            nn.Linear(macro_dim, 32), nn.ReLU(), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, micro: torch.Tensor,
                macro: torch.Tensor) -> torch.Tensor:
        n_nodes = price_seq.shape[0]

        if self.macro_only:
            # No per-ticker input at all: every ticker on this day receives an
            # identical prediction, so this row cannot discriminate stocks.
            self.last_film = None
            return self.macro_head(macro).unsqueeze(0).expand(n_nodes, 2)

        gamma = beta = None
        if self.conditioning in ("micro", "attention"):
            raw = self.film(macro)
            gamma, beta = raw[:self.micro_proj], raw[self.micro_proj:]
            self.last_film = (gamma.detach(), beta.detach())
        else:
            self.last_film = None

        h = self.price_encoder(price_seq, price_mask)

        if self.use_micro:
            projected = self.micro_projection(micro)
            empty = micro.abs().sum(dim=1, keepdim=True) == 0
            projected = torch.where(empty, self.micro_default.expand_as(projected),
                                     projected)
            if self.conditioning == "micro":
                projected = gamma.unsqueeze(0) * projected + beta.unsqueeze(0)
            h = torch.cat([h, projected], dim=1)

        if self.use_graph:
            scale = shift = None
            if self.conditioning == "attention":
                params = self.attention_film(macro)
                scale, shift = params[0], params[1]

            message = torch.relu(self.gat1(h, edge_index))
            if scale is not None:
                message = scale * message + shift
            h = h + message
            h = self.dropout(h)
            message = torch.relu(self.gat2(h, edge_index))
            if scale is not None:
                message = scale * message + shift
            h = h + message

        return self.classifier(h)
