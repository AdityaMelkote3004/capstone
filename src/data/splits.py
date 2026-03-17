"""
Time-Based Train/Validation/Test Splitting

Purpose:
    - Split time series data chronologically
    - Ensure NO future information leakage
    - train < val < test in time

Why time-based splits?
    - Stock data has temporal dependencies
    - Random splits would leak future information
    - We must simulate real forecasting: train on past, predict future
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def split_by_time(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """
    Split data into train/val/test sets chronologically.
    
    Timeline:
        [-----train-----][--val--][--test--]
        older ----------------------> newer
    
    Args:
        X: Input windows, shape (num_samples, window_size, features)
        y: Labels, shape (num_samples,)
        dates: Timestamps for each sample
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        
    Returns:
        Dictionary with keys 'train', 'val', 'test'
        Each value is a tuple: (X_split, y_split, dates_split)
        
    Raises:
        ValueError if ratios don't sum to 1.0
        
    Note:
        - Data is NOT shuffled
        - Order is preserved
        - Earlier data goes to train, later to test
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
    
    n_samples = len(X)
    
    # Compute split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Split the data
    X_train = X[:train_end]
    y_train = y[:train_end]
    dates_train = dates[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    dates_val = dates[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    dates_test = dates[val_end:]
    
    # Print split information
    print(f"\n{'='*60}")
    print(f"Time-based data split")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples}")
    print(f"\nTrain set:")
    print(f"  - Size: {len(X_train)} samples ({train_ratio*100:.1f}%)")
    print(f"  - Date range: {dates_train[0]} to {dates_train[-1]}")
    print(f"  - Input shape: {X_train.shape}")
    print(f"  - Label shape: {y_train.shape}")
    
    print(f"\nValidation set:")
    print(f"  - Size: {len(X_val)} samples ({val_ratio*100:.1f}%)")
    print(f"  - Date range: {dates_val[0]} to {dates_val[-1]}")
    print(f"  - Input shape: {X_val.shape}")
    print(f"  - Label shape: {y_val.shape}")
    
    print(f"\nTest set:")
    print(f"  - Size: {len(X_test)} samples ({test_ratio*100:.1f}%)")
    print(f"  - Date range: {dates_test[0]} to {dates_test[-1]}")
    print(f"  - Input shape: {X_test.shape}")
    print(f"  - Label shape: {y_test.shape}")
    
    # Verify temporal ordering
    print(f"\n{'='*60}")
    print(f"Temporal ordering check:")
    print(f"{'='*60}")
    print(f"Train ends before val? {dates_train[-1] < dates_val[0]}")
    print(f"Val ends before test? {dates_val[-1] < dates_test[0]}")
    
    return {
        'train': (X_train, y_train, dates_train),
        'val': (X_val, y_val, dates_val),
        'test': (X_test, y_test, dates_test)
    }


def split_by_date(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    train_end_date: str,
    val_end_date: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """
    Split data by explicit dates (alternative to ratio-based split).
    
    This is useful when you want to test on a specific time period.
    For example:
        - Train: 2015-2020
        - Val: 2020-2021
        - Test: 2021-2023
    
    Args:
        X: Input windows
        y: Labels
        dates: Timestamps
        train_end_date: Last date for training (exclusive), e.g., "2020-01-01"
        val_end_date: Last date for validation (exclusive), e.g., "2021-01-01"
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)
    
    # Create boolean masks
    train_mask = dates < train_end
    val_mask = (dates >= train_end) & (dates < val_end)
    test_mask = dates >= val_end
    
    # Apply masks
    X_train = X[train_mask]
    y_train = y[train_mask]
    dates_train = dates[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    dates_val = dates[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    dates_test = dates[test_mask]
    
    # Print information
    print(f"\n{'='*60}")
    print(f"Date-based data split")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    
    print(f"\nTrain set (before {train_end_date}):")
    print(f"  - Size: {len(X_train)} samples")
    print(f"  - Date range: {dates_train[0]} to {dates_train[-1]}")
    
    print(f"\nValidation set ({train_end_date} to {val_end_date}):")
    print(f"  - Size: {len(X_val)} samples")
    print(f"  - Date range: {dates_val[0]} to {dates_val[-1]}")
    
    print(f"\nTest set (after {val_end_date}):")
    print(f"  - Size: {len(X_test)} samples")
    print(f"  - Date range: {dates_test[0]} to {dates_test[-1]}")
    
    return {
        'train': (X_train, y_train, dates_train),
        'val': (X_val, y_val, dates_val),
        'test': (X_test, y_test, dates_test)
    }


if __name__ == "__main__":
    # Demo: Create synthetic data and split it
    print("Demo: Time-based splitting")
    
    # Create synthetic data
    n_samples = 1000
    window_size = 20
    
    X = np.random.randn(n_samples, window_size, 1)
    y = np.random.randn(n_samples)
    dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='D')
    
    # Test ratio-based split
    splits = split_by_time(X, y, dates, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"{'='*60}")
