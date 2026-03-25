"""Baseline models: LSTM, Logistic Regression (sklearn), MLP."""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """LSTM with sliding window — sequential model."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, window: torch.Tensor, **kwargs) -> torch.Tensor:
        # window: (batch, W, features)
        _, (h_n, _) = self.lstm(window)
        return self.classifier(h_n[-1])


class MLPBaseline(nn.Module):
    """MLP on flat feature vector (current day)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, flat: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(flat)
