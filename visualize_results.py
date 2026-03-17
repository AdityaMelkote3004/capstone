"""
Visualize Your Stock Prediction Results

This script loads your trained model's predictions and creates visualizations.

Usage:
    python visualize_results.py
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.price_loader import PriceDataLoader, normalize_prices
from src.data.splits import split_by_time
from src.models.price_model import LSTMPriceModel
from src.evaluation.metrics import compute_all_metrics, evaluate_model
from src.visualization.visualize_predictions import (
    load_predictions,
    compute_metrics,
    create_stock_visualization,
    create_detailed_analysis
)
from torch.utils.data import TensorDataset, DataLoader
import yaml


def main():
    """Load results and create visualizations."""
    
    print("="*60)
    print("Stock Prediction Results Visualization")
    print("="*60)
    
    # Load configuration
    config_path = "config/price_baseline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ticker = config['data']['ticker']
    print(f"\nLoading results for: {ticker}")
    
    # ==================== LOAD DATA ====================
    print("\nStep 1: Loading data...")
    loader = PriceDataLoader(
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        window_size=config['data']['window_size']
    )
    
    X, y, dates = loader.load_and_prepare()
    
    # Split data
    print("\nStep 2: Splitting data...")
    splits = split_by_time(
        X, y, dates,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    X_train, y_train, dates_train = splits['train']
    X_val, y_val, dates_val = splits['val']
    X_test, y_test, dates_test = splits['test']
    
    # Normalize
    X_train_norm, X_val_norm, X_test_norm, _ = normalize_prices(
        X_train, X_val, X_test
    )
    
    # ==================== LOAD MODEL ====================
    print("\nStep 3: Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMPriceModel(
        input_dim=1,
        hidden_dim=config['model']['params']['hidden_dim'],
        num_layers=config['model']['params']['num_layers'],
        dropout=config['model']['params']['dropout']
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(
        config['paths']['checkpoints'],
        'best_model.pt'
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    
    # ==================== GENERATE PREDICTIONS ====================
    print("\nStep 4: Generating predictions...")
    
    # Create test dataloader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    predictions, actuals = evaluate_model(model, test_loader, device)
    
    print(f"Generated {len(predictions)} predictions")
    
    # ==================== COMPUTE METRICS ====================
    print("\nStep 5: Computing metrics...")
    metrics = compute_metrics(actuals, predictions)
    
    print(f"\nTest Set Metrics:")
    print(f"  MSE:                 {metrics['mse']:.6f}")
    print(f"  MAE:                 {metrics['mae']:.6f} ({metrics['mae']*100:.2f}%)")
    print(f"  RMSE:                {metrics['rmse']:.6f} ({metrics['rmse']*100:.2f}%)")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.4f} ({metrics['directional_accuracy']*100:.2f}%)")
    
    # ==================== CREATE VISUALIZATIONS ====================
    print("\nStep 6: Creating visualizations...")
    
    # Prepare DataFrame
    from src.visualization.visualize_predictions import load_predictions
    df = load_predictions(predictions, actuals, dates_test)
    
    # Output directory
    viz_dir = "results/price_only/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create simple visualization
    print("  - Creating main visualization...")
    create_stock_visualization(
        df,
        metrics,
        ticker=ticker,
        save_path=os.path.join(viz_dir, f"{ticker}_prediction_visualization.png")
    )
    
    # Create detailed analysis
    print("  - Creating detailed analysis...")
    create_detailed_analysis(
        df,
        metrics,
        ticker=ticker,
        save_path=os.path.join(viz_dir, f"{ticker}_detailed_analysis.png")
    )
    
    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print(f"{'='*60}")
    print(f"\nVisualization files saved to: {viz_dir}")
    print(f"  1. {ticker}_prediction_visualization.png")
    print(f"  2. {ticker}_detailed_analysis.png")
    print(f"\nOpening visualizations...")
    
    # Open the files
    import subprocess
    subprocess.run(['start', os.path.join(viz_dir, f"{ticker}_prediction_visualization.png")], shell=True)
    subprocess.run(['start', os.path.join(viz_dir, f"{ticker}_detailed_analysis.png")], shell=True)


if __name__ == "__main__":
    main()
