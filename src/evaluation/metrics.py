"""
Evaluation Metrics for Stock Return Prediction

Purpose:
    - Compute standard regression metrics
    - Compute financial metrics (directional accuracy)
    - Provide clear evaluation of model performance

Metrics:
    1. MSE (Mean Squared Error): Standard regression metric
    2. MAE (Mean Absolute Error): Robust to outliers
    3. Directional Accuracy: Percent of correct sign predictions
       (This is what matters for trading!)
"""

import numpy as np
import torch
from typing import Dict, Tuple


def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Squared Error.
    
    MSE = (1/n) * sum((pred - target)^2)
    
    Args:
        predictions: Predicted returns
        targets: True returns
        
    Returns:
        MSE value (lower is better)
    """
    return np.mean((predictions - targets) ** 2)


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = (1/n) * sum(|pred - target|)
    
    Args:
        predictions: Predicted returns
        targets: True returns
        
    Returns:
        MAE value (lower is better)
    """
    return np.mean(np.abs(predictions - targets))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt(MSE)
    
    Same units as the target variable (returns).
    
    Args:
        predictions: Predicted returns
        targets: True returns
        
    Returns:
        RMSE value (lower is better)
    """
    return np.sqrt(compute_mse(predictions, targets))


def compute_directional_accuracy(
    predictions: np.ndarray, 
    targets: np.ndarray
) -> float:
    """
    Compute directional accuracy.
    
    What it measures:
        - Percent of times we correctly predicted the SIGN of return
        - sign(pred) == sign(target)
    
    Why it matters:
        - In trading, we care about direction (up/down)
        - If we predict direction correctly, we can make money
        - More important than exact magnitude
    
    Args:
        predictions: Predicted returns
        targets: True returns
        
    Returns:
        Directional accuracy in [0, 1] (higher is better)
        0.5 = random guessing
        1.0 = perfect direction prediction
    """
    # Get signs (positive, negative, or zero)
    pred_signs = np.sign(predictions)
    target_signs = np.sign(targets)
    
    # Count correct predictions
    correct = np.sum(pred_signs == target_signs)
    total = len(predictions)
    
    return correct / total


def compute_all_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted returns (1D array)
        targets: True returns (1D array)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'mse': compute_mse(predictions, targets),
        'mae': compute_mae(predictions, targets),
        'rmse': compute_rmse(predictions, targets),
        'directional_accuracy': compute_directional_accuracy(predictions, targets)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = ""):
    """
    Print metrics in a readable format.
    
    Args:
        metrics: Dictionary of metric values
        dataset_name: Name of dataset (e.g., "Test")
    """
    title = f"{dataset_name} Metrics" if dataset_name else "Metrics"
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"MSE:                   {metrics['mse']:.6f}")
    print(f"MAE:                   {metrics['mae']:.6f}")
    print(f"RMSE:                  {metrics['rmse']:.6f}")
    print(f"Directional Accuracy:  {metrics['directional_accuracy']:.4f} ({metrics['directional_accuracy']*100:.2f}%)")
    
    # Interpretation
    print(f"\nInterpretation:")
    if metrics['directional_accuracy'] > 0.55:
        print(f"  ✓ Direction prediction is BETTER than random (>50%)")
    elif metrics['directional_accuracy'] > 0.50:
        print(f"  ~ Direction prediction is SLIGHTLY better than random")
    else:
        print(f"  ✗ Direction prediction is NO BETTER than random (≤50%)")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for an entire dataset.
    
    This is a helper function for evaluation.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with evaluation data
        device: Device to run on (cuda/cpu)
        
    Returns:
        predictions: All predictions (1D array)
        targets: All true labels (1D array)
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for batch_X, batch_y in dataloader:
        # Move to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        
        # Store results
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    return predictions, targets


if __name__ == "__main__":
    # Demo: Test metrics with synthetic data
    print("Demo: Testing evaluation metrics")
    
    # Create synthetic predictions and targets
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate returns (centered around 0, small values)
    true_returns = np.random.randn(n_samples) * 0.02
    
    # Case 1: Perfect predictions
    print("\n" + "="*60)
    print("Case 1: Perfect Predictions")
    print("="*60)
    perfect_preds = true_returns.copy()
    metrics = compute_all_metrics(perfect_preds, true_returns)
    print_metrics(metrics, "Perfect Model")
    
    # Case 2: Noisy predictions (but correlated)
    print("\n" + "="*60)
    print("Case 2: Noisy but Correlated Predictions")
    print("="*60)
    noisy_preds = true_returns + np.random.randn(n_samples) * 0.01
    metrics = compute_all_metrics(noisy_preds, true_returns)
    print_metrics(metrics, "Noisy Model")
    
    # Case 3: Random predictions
    print("\n" + "="*60)
    print("Case 3: Random Predictions (Baseline)")
    print("="*60)
    random_preds = np.random.randn(n_samples) * 0.02
    metrics = compute_all_metrics(random_preds, true_returns)
    print_metrics(metrics, "Random Model")
    
    # Case 4: Opposite predictions (worst case)
    print("\n" + "="*60)
    print("Case 4: Opposite Predictions (Worst Case)")
    print("="*60)
    opposite_preds = -true_returns
    metrics = compute_all_metrics(opposite_preds, true_returns)
    print_metrics(metrics, "Opposite Model")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
