"""GRU + temporal-attention price encoder.

Reused unmodified from Experiment 3's MAN-SF reimplementation
(preprocessing_notebooks/MMGTFFF_mansf_joint_bilinear.ipynb), which itself
is built from the architecture description in Sawhney et al. (EMNLP 2020,
"Deep Attentive Learning for Stock Movement Prediction From Social Media
Text and Company Correlations") -- given its own module here since
Experiments 3 and 4 both depend on it.
"""

import torch
import torch.nn as nn


class PriceEncoder(nn.Module):
    """Encodes a masked, variable-length price window into a fixed vector."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.temporal_score = nn.Linear(hidden_dim, 1)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, input_dim); mask: (batch, T), 1 = real day, 0 = padding
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, total_length=x.shape[1]
        )
        scores = self.temporal_score(gru_out).squeeze(-1).masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * gru_out).sum(dim=1)
