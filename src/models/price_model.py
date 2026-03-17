"""
Baseline LSTM Model for Stock Return Prediction

Purpose:
    - Simple LSTM architecture
    - Input: sequence of past prices
    - Output: single scalar (predicted next-day return)
    - This is a REGRESSION task, not classification

Architecture:
    - LSTM layers to capture temporal patterns
    - Fully connected layer to produce scalar output
    - No softmax/sigmoid (we predict continuous returns)
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMPriceModel(nn.Module):
    """
    LSTM-based model for predicting next-day stock returns.
    
    Architecture:
        Input: (batch_size, window_size, input_dim)
            - window_size: number of past days (W)
            - input_dim: number of features per day (1 for price only)
        
        LSTM: processes sequence, captures temporal patterns
        
        Output: (batch_size, 1)
            - Single scalar per sample
            - Predicted next-day return
    
    Why LSTM?
        - Designed for sequential data
        - Can capture long-term dependencies
        - Standard baseline for time series
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Number of features per time step (1 for price only)
            hidden_dim: Size of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between LSTM layers
                    (helps prevent overfitting)
        """
        super(LSTMPriceModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer(s)
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer to produce scalar output
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
            
        Returns:
            predictions: Tensor of shape (batch_size, 1)
                        Predicted next-day returns
        """
        # x shape: (batch_size, window_size, input_dim)
        
        # Pass through LSTM
        # lstm_out: (batch_size, window_size, hidden_dim)
        # hidden: (num_layers, batch_size, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last time step's output
        # last_output: (batch_size, hidden_dim)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        # prediction: (batch_size, 1)
        prediction = self.fc(last_output)
        
        return prediction
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPPriceModel(nn.Module):
    """
    Alternative: Simple MLP baseline (flattens the window).
    
    This is included as a simpler alternative to LSTM.
    Sometimes a simple MLP can be competitive!
    
    Architecture:
        - Flatten input window
        - Pass through fully connected layers
        - Output single scalar
    """
    
    def __init__(
        self,
        window_size: int = 20,
        input_dim: int = 1,
        hidden_dims: list = [128, 64],
        dropout: float = 0.2
    ):
        """
        Initialize the MLP model.
        
        Args:
            window_size: Number of time steps in input
            input_dim: Number of features per time step
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(MLPPriceModel, self).__init__()
        
        self.window_size = window_size
        self.input_dim = input_dim
        
        # Input dimension after flattening
        flattened_dim = window_size * input_dim
        
        # Build layers
        layers = []
        prev_dim = flattened_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, window_size, input_dim)
            
        Returns:
            predictions: Tensor (batch_size, 1)
        """
        # Flatten the input
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Pass through network
        prediction = self.network(x_flat)
        
        return prediction
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str = "lstm",
    window_size: int = 20,
    input_dim: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: Type of model ("lstm" or "mlp")
        window_size: Number of time steps in input
        input_dim: Number of features per time step
        **kwargs: Additional model-specific parameters
        
    Returns:
        Initialized model
    """
    if model_type.lower() == "lstm":
        model = LSTMPriceModel(
            input_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', 64),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.2)
        )
    elif model_type.lower() == "mlp":
        model = MLPPriceModel(
            window_size=window_size,
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [128, 64]),
            dropout=kwargs.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\n{'='*60}")
    print(f"Model created: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Demo: Create models and test forward pass
    print("Demo: Testing model architectures")
    
    # Create synthetic input
    batch_size = 32
    window_size = 20
    input_dim = 1
    
    x = torch.randn(batch_size, window_size, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Test LSTM model
    print("\n" + "="*60)
    print("Testing LSTM model")
    print("="*60)
    lstm_model = create_model("lstm", window_size=window_size)
    lstm_output = lstm_model(x)
    print(f"Output shape: {lstm_output.shape}")
    print(f"Expected: ({batch_size}, 1)")
    
    # Test MLP model
    print("\n" + "="*60)
    print("Testing MLP model")
    print("="*60)
    mlp_model = create_model("mlp", window_size=window_size)
    mlp_output = mlp_model(x)
    print(f"Output shape: {mlp_output.shape}")
    print(f"Expected: ({batch_size}, 1)")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
