"""Experiment 8's model: a 4-day LSTM over price/count features plus an
OPTIONAL low-dimensional static text vector.

The design follows one empirical result from Experiments 1-7: across four
architectures, tweet COUNTS (one number) beat mean-pooled FinBERT semantics
(768 numbers) every time -- +0.092 vs +0.036 pooled MCC on the same test split.

Two mechanisms explain that, and this model is built to avoid both:

  1. Mean-pooling 64 tweets averages a handful of informative ones into a
     generic "financial text" centroid. Counts survive because a count cannot
     be diluted by averaging.
  2. Going from ~15 inputs to 1536 multiplies free parameters against a fixed
     20,344 training samples, so the extra capacity fits noise.

So text enters here as a handful of dimensions, never as the raw 768, and the
count features are kept rather than replaced.
"""

import torch
import torch.nn as nn


class LowDimTextLSTM(nn.Module):
    """LSTM over `(batch, 4, per_day_dim)` plus `(batch, text_dim)` appended
    after the recurrence.

    Text is concatenated post-LSTM rather than per timestep because the cached
    tweet vector describes the whole lookback window, not any single day of it
    -- broadcasting it across timesteps would imply a per-day resolution the
    feature does not have.
    """

    def __init__(self, per_day_dim: int, text_dim: int = 0, hidden: int = 32,
                 dropout: float = 0.3, layers: int = 1):
        super().__init__()
        self.text_dim = text_dim
        self.lstm = nn.LSTM(per_day_dim, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + text_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, lengths: torch.Tensor,
                text: torch.Tensor = None) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            price_seq, lengths.cpu().clamp(min=1), batch_first=True,
            enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        features = hidden[-1]
        if self.text_dim:
            if text is None:
                raise ValueError("model was built with text_dim>0 but got no text")
            features = torch.cat([features, text], dim=1)
        return self.head(features)
