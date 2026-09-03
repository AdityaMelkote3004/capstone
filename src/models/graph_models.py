"""GraphSAGE and R-GCN variants for Experiment 6's dense stock graph.

Both wrap each message-passing layer in a residual connection, for the same
reason StockGAT does (see results/experiment4_gat/EXPERIMENT4_GAT.md): the
sector graph is a set of complete cliques, and plain stacked message
passing over a complete graph collapses every node to an identical output.

R-GCN is the only model here that learns a separate transformation per
edge type, which is why it is included -- it is the natural fit for a graph
whose edges mean three different things.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, SAGEConv

from src.models.price_encoder import PriceEncoder


class StockGraphSAGE(nn.Module):
    """PriceEncoder per ticker -> 2 residual SAGEConv layers -> classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        h = self.price_encoder(price_seq, price_mask)
        h = h + torch.relu(self.conv1(h, edge_index))
        h = self.dropout(h)
        h = h + torch.relu(self.conv2(h, edge_index))
        return self.classifier(h)


class StockRGCN(nn.Module):
    """PriceEncoder per ticker -> 2 residual RGCNConv layers -> classifier.

    `num_relations` is always 3 even when the active edge combination uses
    fewer; the unused relation's weights simply receive no gradient. This
    keeps one model definition across all 7 edge combinations.
    """

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 num_relations: int = 3, dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        h = self.price_encoder(price_seq, price_mask)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long,
                                    device=edge_index.device)
        h = h + torch.relu(self.conv1(h, edge_index, edge_type))
        h = self.dropout(h)
        h = h + torch.relu(self.conv2(h, edge_index, edge_type))
        return self.classifier(h)
