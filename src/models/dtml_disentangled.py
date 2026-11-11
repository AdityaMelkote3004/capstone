"""MMGT-D: MMGT with market timing and stock selection disentangled (Experiment 12).

MMGT adds the market summary to every stock's context before stock attention, so
market information and stock-specific information are mixed in one representation.
MMGT-D separates them structurally:

  market branch   the market-index encoder alone gives one day-level logit m_t,
                  trained to predict the market's direction (auxiliary loss).
  stock branch    stock encoder + multi-modal profile + stock attention, with NO
                  market input. Its final states are centred over the day's
                  window-valid stocks and passed to a bias-free linear head, so the
                  excess logits e_{i,t} sum to exactly zero every day: they can only
                  rank stocks against each other, never call the day.

  mode "struct":  logit = m_t + e_{i,t}
  mode "beta":    logit = (1 + b_i) m_t + e_{i,t},  b_i = tanh(w . h_i): each stock's
                  systematic response to the market (market / systematic /
                  idiosyncratic three-way split).
  adversary:      optional gradient-reversal head (ADD-style) that tries to predict
                  the market's direction from permutation-invariant statistics of the
                  centred stock states; the stock branch is trained to defeat it.

Centring uses window-valid stocks, never labelled ones: which stocks are labelled on
day t depends on day t's move.
"""

import torch
import torch.nn as nn

from src.models.dtml import DTML


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


class DTMLDisentangled(DTML):
    def __init__(self, *args, mode: str = "struct", adversary: bool = False,
                 adv_weight: float = 1.0, **kwargs):
        kwargs["use_mc"] = False                      # market never enters the stock branch
        super().__init__(*args, **kwargs)
        h = self.output.in_features
        self.mode, self.adv_weight = mode, adv_weight
        del self.output
        self.excess_head = nn.Linear(h, 1, bias=False)
        self.market_head = nn.Linear(h, 1)
        self.beta_head = nn.Linear(h, 1) if mode == "beta" else None
        self.adversary = (nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
                          if adversary else None)
        self.parts = {}

    def forward(self, x, valid, market, extra=None, adj=None):
        context = self.stock_encoder(x)
        present = context[valid]
        context = (self.norm_scale * (context - present.mean()) /
                   (present.std() + 1e-6) + self.norm_shift)
        if self.extra is not None and extra is not None:
            context = context + self.extra(extra)
        hidden = context
        if self.use_sa:
            mixed, _ = self.attention(context.unsqueeze(0), context.unsqueeze(0),
                                      context.unsqueeze(0),
                                      key_padding_mask=~valid.unsqueeze(0))
            combined = context + self.dropout(mixed.squeeze(0))
            hidden = self.out_norm(torch.tanh(combined + self.mlp(combined)))

        centred = hidden - hidden[valid].mean(0, keepdim=True)
        e = self.excess_head(centred).squeeze(-1)                       # S
        hm = self.market_encoder(market.unsqueeze(0))                   # 1, h
        m = self.market_head(self.dropout(hm)).squeeze()                # scalar
        if self.mode == "beta":
            b = torch.tanh(self.beta_head(hidden).squeeze(-1))
            logit = (1 + b) * m + e
        else:
            b = torch.zeros_like(e)
            logit = m + e
        adv = None
        if self.adversary is not None:
            z = _GradReverse.apply(centred[valid], self.adv_weight)
            stats = torch.cat([z.abs().mean(0), z.std(0)])
            adv = self.adversary(stats).squeeze()
        self.parts = {"m": m.expand_as(e), "e": e, "b": b, "adv": adv}
        return logit

    def selective_penalty(self) -> torch.Tensor:
        pen = self.excess_head.weight.pow(2).sum()
        pen = pen + self.market_head.weight.pow(2).sum() + self.market_head.bias.pow(2).sum()
        if self.beta_head is not None:
            pen = pen + self.beta_head.weight.pow(2).sum() + self.beta_head.bias.pow(2).sum()
        return pen
