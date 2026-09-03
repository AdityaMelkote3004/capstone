"""Graph transformer with one learned attention bias per edge relation.

Generalises Experiment 5's StockGraphTransformer (which had a single
learned scalar for "same sector") to a graph whose edges mean three
different things. Experiment 5's model is left untouched in
src/models/graph_transformer.py so its published results stay reproducible.

Attention is global -- every ticker attends to every other ticker in the
day's cross-section -- and the graph enters as an additive bias on the raw
attention scores, the Graphormer-style mechanism nn.TransformerEncoderLayer
supports natively via a float src_mask.
"""

import torch
import torch.nn as nn

from src.models.price_encoder import PriceEncoder


class MultiRelationGraphTransformer(nn.Module):
    """PriceEncoder -> 2 global-attention layers biased per relation ->
    per-ticker classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 nhead: int = 4, dim_feedforward: int = 128,
                 dropout: float = 0.2, num_relations: int = 3):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.relation_bias = nn.Parameter(torch.zeros(num_relations))
        layer_kwargs = dict(d_model=hidden_dim, nhead=nhead,
                            dim_feedforward=dim_feedforward, dropout=dropout,
                            batch_first=True)
        self.layer1 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.layer2 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def _bias_matrix(self, n: int, edge_index: torch.Tensor,
                     edge_type: torch.Tensor) -> torch.Tensor:
        """(N,N) additive attention bias: entry [i,j] is the learned scalar
        for the relation connecting i and j, or 0 where they are unconnected."""
        bias = torch.zeros(n, n, device=self.relation_bias.device)
        if edge_index.numel() == 0:
            return bias
        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long,
                                    device=edge_index.device)
        bias[edge_index[0], edge_index[1]] = self.relation_bias[edge_type]
        return bias

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        h = self.price_encoder(price_seq, price_mask)
        bias = self._bias_matrix(h.shape[0], edge_index, edge_type)
        x = h.unsqueeze(0)                       # (1, N, hidden) for batch_first
        x = self.layer1(x, src_mask=bias)
        x = self.layer2(x, src_mask=bias)
        return self.classifier(x.squeeze(0))
