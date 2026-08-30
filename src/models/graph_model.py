"""Sector graph construction and the sector-graph GAT model (Experiment 4).

The sector graph connects tickers that share a `Sector` label -- a cheaper,
materially different proxy for MAN-SF's Wikidata company-relation graph,
NOT a reproduction of it. See
docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md for why
this experiment's results must not be compared directly to MAN-SF's
GAT+price ablation figure (0.117 MCC, measured on their Wikidata graph).
"""

from itertools import combinations

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from src.models.price_encoder import PriceEncoder


def build_sector_graph(sector_csv_path: str) -> nx.Graph:
    """Builds an undirected graph: one node per ticker, edges between
    tickers in the same sector. Node attributes: sector, name."""
    df = pd.read_csv(sector_csv_path)
    graph = nx.Graph()
    for _, row in df.iterrows():
        graph.add_node(row["Ticker"], sector=row["Sector"], name=row["Ticker"])
    for _, group in df.groupby("Sector"):
        for ticker_a, ticker_b in combinations(sorted(group["Ticker"]), 2):
            graph.add_edge(ticker_a, ticker_b)
    return graph


def edge_index_for_tickers(graph: nx.Graph, tickers: list) -> torch.Tensor:
    """Builds a (2, E) edge_index over `tickers`, using each ticker's
    position in the list as its node index, restricted to `graph` edges
    where both endpoints are in `tickers`. Returns both directions per
    undirected edge (required by GATConv's message passing)."""
    index_of = {ticker: i for i, ticker in enumerate(tickers)}
    ticker_set = set(tickers)
    src, dst = [], []
    for a, b in graph.edges():
        if a in ticker_set and b in ticker_set:
            src.extend([index_of[a], index_of[b]])
            dst.extend([index_of[b], index_of[a]])
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


class StockGAT(nn.Module):
    """Price-only sector-graph GAT: PriceEncoder per ticker -> 2-layer
    GATConv over the sector graph -> per-ticker binary classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=False, dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=False, dropout=dropout, add_self_loops=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        node_features = self.price_encoder(price_seq, price_mask)
        h = torch.relu(self.gat1(node_features, edge_index))
        h = self.dropout(h)
        h = torch.relu(self.gat2(h, edge_index))
        return self.classifier(h)
