"""DTML: Data-axis Transformer with Multi-Level contexts (Yoo et al., KDD 2021).

Reported on ACL18: ACC 0.5744 +/- 0.0194, MCC 0.1910 +/- 0.0315 over 10 seeds --
roughly double anything this project has measured on the same test window.
This module follows the paper's Sections 3.2-3.5:

  3.2  per-stock attention LSTM over a w-day window (shared weights), with a
       tanh feature transform first and "context normalisation" after: a
       layer-norm variant whose mean/std run over all stocks and elements, with
       a learnable scale and shift per (stock, element).
  3.3  multi-level context  h_m = h_c + beta * h_market, where h_market is the
       same attention LSTM applied to the market index series.
  3.4  one multi-head self-attention layer ACROSS STOCKS (the "data axis"),
       then  H_p = tanh(H + H~ + MLP(H + H~))  with dropout and layer norm.
  3.5  sigmoid output, cross-entropy, and selective L2 on the output layer only.

The use_ta / use_sa / use_mc switches reproduce the paper's own ablation table
(DTML-TA, DTML-SA, DTML-MC), so its claims can be checked component by component.

Note on use_mc: the market term is identical for every stock on a day, so it can
only move all of a day's predictions together. It is a market-timing mechanism
by construction. On a 64-day test set, pooled MCC rewards that heavily, which is
why every run here also reports within-day MCC and a day-collapse control.
"""

import torch
import torch.nn as nn


class AttentionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, use_ta: bool = True):
        super().__init__()
        self.transform = nn.Linear(n_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.use_ta = use_ta

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, W, F) -> (B, h)
        states, _ = self.lstm(torch.tanh(self.transform(x)))
        last = states[:, -1]
        if not self.use_ta:
            return last
        scores = torch.softmax((states * last.unsqueeze(1)).sum(-1), dim=1)
        return (scores.unsqueeze(-1) * states).sum(1)


class DTML(nn.Module):
    def __init__(self, n_stocks: int, n_features: int = 11, hidden: int = 64,
                 heads: int = 4, beta: float = 0.1, dropout: float = 0.15,
                 use_ta: bool = True, use_sa: bool = True, use_mc: bool = True,
                 extra_dim: int = 0):
        super().__init__()
        self.beta, self.use_sa, self.use_mc = beta, use_sa, use_mc
        self.stock_encoder = AttentionLSTM(n_features, hidden, use_ta)
        self.market_encoder = AttentionLSTM(n_features, hidden, use_ta)
        self.norm_scale = nn.Parameter(torch.ones(n_stocks, hidden))
        self.norm_shift = nn.Parameter(torch.zeros(n_stocks, hidden))
        # Optional per-stock static features (Exp 10's volatility / fundamentals /
        # counts / peer block) added to the context. Off for the faithful rung.
        self.extra = nn.Linear(extra_dim, hidden) if extra_dim else None
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                               batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.ReLU(),
                                 nn.Linear(4 * hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, market: torch.Tensor,
                extra: torch.Tensor = None) -> torch.Tensor:
        """x (S, W, F) one day's windows; valid (S,) stocks with complete windows;
        market (W, F). Returns one logit per stock."""
        context = self.stock_encoder(x)                                   # S, h
        present = context[valid]
        context = (self.norm_scale * (context - present.mean()) /
                   (present.std() + 1e-6) + self.norm_shift)
        if self.extra is not None and extra is not None:
            context = context + self.extra(extra)
        if self.use_mc:
            context = context + self.beta * self.market_encoder(market.unsqueeze(0))

        hidden = context
        if self.use_sa:
            mixed, _ = self.attention(context.unsqueeze(0), context.unsqueeze(0),
                                      context.unsqueeze(0),
                                      key_padding_mask=~valid.unsqueeze(0))
            combined = context + self.dropout(mixed.squeeze(0))
            hidden = self.out_norm(torch.tanh(combined + self.mlp(combined)))
        return self.output(hidden).squeeze(-1)

    def selective_penalty(self) -> torch.Tensor:
        """Section 3.5: L2 on the final predictor only, leaving the encoders free."""
        return self.output.weight.pow(2).sum() + self.output.bias.pow(2).sum()
