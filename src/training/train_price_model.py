"""
Training Script for Price-Based Stock Return Prediction

Purpose:
    - Complete training pipeline
    - Train on historical prices only
    - Predict next-day returns
    - Evaluate on test set

This is the main executable script.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.price_loader import PriceDataLoader, normalize_prices
from src.data.splits import split_by_time
from src.models.price_model import create_model
from src.evaluation.metrics import compute_all_metrics, print_metrics, evaluate_model
from src.utils.seed import set_seed


class Trainer:
    """
    Trainer class for stock return prediction.
    
    Handles:
        - Training loop
        - Validation
        - Checkpointing
        - Logging
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Dictionary with all hyperparameters
        """
        self.config = config
        
        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config['training']['use_gpu'] 
            else 'cpu'
        )
        print(f"\nUsing device: {self.device}")
        
        # Initialize model
        self.model = create_model(
            model_type=config['model']['type'],
            window_size=config['data']['window_size'],
            input_dim=1,
            **config['model'].get('params', {})
        )
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0)
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average training loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            # Move to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)  # (batch_size, 1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_X)
            
            # Compute loss
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)
            
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        num_epochs: int
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"  → New best model saved (val_loss: {val_loss:.6f})")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        save_dir = self.config['paths']['checkpoints']
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        path = os.path.join(save_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config['paths']['checkpoints'], filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {path}")
    
    def plot_losses(self, save_path: str = None):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        
        plt.close()


def main(config_path: str):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"Stock Return Prediction - Training")
    print(f"{'='*60}")
    print(f"Configuration loaded from: {config_path}")
    
    # Set random seed for reproducibility
    set_seed(config['training']['seed'])
    
    # ==================== DATA LOADING ====================
    print(f"\n{'='*60}")
    print(f"STEP 1: Loading Data")
    print(f"{'='*60}")
    
    loader = PriceDataLoader(
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        window_size=config['data']['window_size']
    )
    
    X, y, dates = loader.load_and_prepare()
    
    # ==================== TRAIN/VAL/TEST SPLIT ====================
    print(f"\n{'='*60}")
    print(f"STEP 2: Splitting Data")
    print(f"{'='*60}")
    
    splits = split_by_time(
        X, y, dates,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    X_train, y_train, dates_train = splits['train']
    X_val, y_val, dates_val = splits['val']
    X_test, y_test, dates_test = splits['test']
    
    # ==================== NORMALIZATION ====================
    print(f"\n{'='*60}")
    print(f"STEP 3: Normalizing Data")
    print(f"{'='*60}")
    
    X_train_norm, X_val_norm, X_test_norm, (mean, std) = normalize_prices(
        X_train, X_val, X_test
    )
    
    # ==================== CREATE DATALOADERS ====================
    print(f"\n{'='*60}")
    print(f"STEP 4: Creating DataLoaders")
    print(f"{'='*60}")
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    print(f"DataLoaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    # ==================== TRAINING ====================
    print(f"\n{'='*60}")
    print(f"STEP 5: Training Model")
    print(f"{'='*60}")
    
    trainer = Trainer(config)
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    # Plot training curves
    results_dir = config['paths']['results']
    os.makedirs(results_dir, exist_ok=True)
    trainer.plot_losses(os.path.join(results_dir, 'loss_curve.png'))
    
    # ==================== EVALUATION ====================
    print(f"\n{'='*60}")
    print(f"STEP 6: Final Evaluation")
    print(f"{'='*60}")
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Evaluate on all sets
    for split_name, loader in [('Train', train_loader), 
                                ('Validation', val_loader), 
                                ('Test', test_loader)]:
        predictions, targets = evaluate_model(trainer.model, loader, trainer.device)
        metrics = compute_all_metrics(predictions, targets)
        print_metrics(metrics, split_name)
    
    # Save test metrics
    test_predictions, test_targets = evaluate_model(trainer.model, test_loader, trainer.device)
    test_metrics = compute_all_metrics(test_predictions, test_targets)
    
    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Stock Return Prediction - Test Set Metrics\n")
        f.write(f"{'='*60}\n")
        f.write(f"Ticker: {config['data']['ticker']}\n")
        f.write(f"Test Period: {dates_test[0]} to {dates_test[-1]}\n")
        f.write(f"\n")
        f.write(f"MSE:                   {test_metrics['mse']:.6f}\n")
        f.write(f"MAE:                   {test_metrics['mae']:.6f}\n")
        f.write(f"RMSE:                  {test_metrics['rmse']:.6f}\n")
        f.write(f"Directional Accuracy:  {test_metrics['directional_accuracy']:.4f}\n")
    
    print(f"\nMetrics saved to {metrics_path}")
    
    print(f"\n{'='*60}")
    print(f"Training pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    # Default config path
    config_path = "config/price_baseline.yaml"
    
    # Allow command-line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    main(config_path)
