"""
Price Data Loader for Stock Return Prediction

Purpose:
    - Downloads historical stock price data from Yahoo Finance
    - Creates sliding windows for time series prediction
    - Computes next-day returns as labels
    - Handles missing data appropriately

NO future information leakage.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime


class PriceDataLoader:
    """
    Loads and preprocesses stock price data for return prediction.
    
    This class handles:
    1. Downloading data from Yahoo Finance
    2. Creating sliding windows from price history
    3. Computing next-day returns as labels
    4. Ensuring no future information leakage
    """
    
    def __init__(
        self, 
        ticker: str,
        start_date: str,
        end_date: str,
        window_size: int = 20
    ):
        """
        Initialize the price data loader.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            window_size: Number of days to use as input (W)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.prices_df = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download historical price data from Yahoo Finance.
        
        Returns:
            DataFrame with Date index and Close price column
            
        Prints:
            - Date range
            - Number of data points
            - Sample of data
        """
        print(f"\n{'='*60}")
        print(f"Downloading data for {self.ticker}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"{'='*60}")
        
        # Download data using yfinance
        data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Keep only the Close price
        if 'Close' not in data.columns:
            raise ValueError(f"No 'Close' column found for {self.ticker}")
        
        prices_df = data[['Close']].copy()
        prices_df.columns = ['close']
        
        # Sort by date to ensure temporal order
        prices_df = prices_df.sort_index()
        
        # Handle missing values by forward filling
        # This is reasonable for financial data (assumes price unchanged if missing)
        if prices_df.isnull().any().any():
            print(f"Warning: Found {prices_df.isnull().sum().sum()} missing values")
            print("Forward filling missing values...")
            prices_df = prices_df.fillna(method='ffill')
            
            # If still have NaN at the beginning, use backward fill
            prices_df = prices_df.fillna(method='bfill')
        
        print(f"\nData downloaded successfully!")
        print(f"Total data points: {len(prices_df)}")
        print(f"Date range: {prices_df.index[0]} to {prices_df.index[-1]}")
        print(f"\nFirst 5 rows:")
        print(prices_df.head())
        print(f"\nLast 5 rows:")
        print(prices_df.tail())
        
        self.prices_df = prices_df
        return prices_df
    
    def compute_returns(self) -> pd.DataFrame:
        """
        Compute next-day returns.
        
        return_t = (price_t - price_{t-1}) / price_{t-1}
        
        This is the TARGET we want to predict.
        
        Returns:
            DataFrame with prices and returns
        """
        if self.prices_df is None:
            raise ValueError("Must call download_data() first")
        
        df = self.prices_df.copy()
        
        # Compute next-day return
        # We shift(-1) to get tomorrow's price, then compute return
        df['next_day_return'] = (df['close'].shift(-1) - df['close']) / df['close']
        
        # Drop the last row (no next day return for the last day)
        df = df.dropna()
        
        print(f"\nReturns computed!")
        print(f"Data points with labels: {len(df)}")
        print(f"\nReturn statistics:")
        print(df['next_day_return'].describe())
        
        return df
    
    def create_sliding_windows(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Create sliding windows for time series prediction.
        
        For each position t:
            - Input: prices from [t-W+1, ..., t]
            - Label: return at t+1
        
        Args:
            df: DataFrame with 'close' and 'next_day_return' columns
            
        Returns:
            X: array of shape (num_samples, window_size, 1)
               Contains price windows
            y: array of shape (num_samples,)
               Contains next-day returns
            dates: DatetimeIndex of prediction dates (t+1)
               
        Note:
            - Each window uses ONLY past data
            - No future information leakage
            - Windows are overlapping (standard practice)
        """
        prices = df['close'].values
        returns = df['next_day_return'].values
        dates = df.index
        
        # We need at least window_size points to create one window
        if len(prices) < self.window_size:
            raise ValueError(
                f"Not enough data. Need at least {self.window_size} points, "
                f"but got {len(prices)}"
            )
        
        X_list = []
        y_list = []
        date_list = []
        
        # Create sliding windows
        # Start from window_size-1 because we need W past points
        for i in range(self.window_size - 1, len(prices)):
            # Input: window of past W prices
            window_start = i - self.window_size + 1
            window_end = i + 1
            price_window = prices[window_start:window_end]
            
            # Label: next-day return
            return_label = returns[i]
            
            X_list.append(price_window)
            y_list.append(return_label)
            date_list.append(dates[i])
        
        # Convert to numpy arrays
        X = np.array(X_list)  # shape: (num_samples, window_size)
        y = np.array(y_list)  # shape: (num_samples,)
        
        # Reshape X to (num_samples, window_size, 1)
        # This adds a feature dimension (useful for LSTM/CNN)
        X = X.reshape(-1, self.window_size, 1)
        
        print(f"\n{'='*60}")
        print(f"Sliding windows created!")
        print(f"{'='*60}")
        print(f"Window size (W): {self.window_size}")
        print(f"Number of samples: {len(X)}")
        print(f"Input shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Date range: {date_list[0]} to {date_list[-1]}")
        
        return X, y, pd.DatetimeIndex(date_list)
    
    def load_and_prepare(self) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Complete pipeline: download -> compute returns -> create windows.
        
        Returns:
            X: Input windows (num_samples, window_size, 1)
            y: Return labels (num_samples,)
            dates: Dates for each sample
        """
        # Step 1: Download data
        df = self.download_data()
        
        # Step 2: Compute returns
        df = self.compute_returns()
        
        # Step 3: Create sliding windows
        X, y, dates = self.create_sliding_windows(df)
        
        return X, y, dates


def normalize_prices(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None
) -> Tuple:
    """
    Normalize price windows using statistics from training set ONLY.
    
    Why normalize?
    - Neural networks train better with normalized inputs
    - Different stocks have different price scales
    
    Method:
    - Compute mean and std from training set
    - Apply same normalization to val/test sets
    - NO information leakage
    
    Args:
        X_train: Training windows (n_train, W, 1)
        X_val: Validation windows (n_val, W, 1) or None
        X_test: Test windows (n_test, W, 1) or None
        
    Returns:
        Normalized versions of inputs, plus (mean, std) statistics
    """
    # Compute statistics from training set only
    train_mean = X_train.mean()
    train_std = X_train.std()
    
    print(f"\nNormalization statistics (from training set):")
    print(f"Mean: {train_mean:.2f}")
    print(f"Std: {train_std:.2f}")
    
    # Normalize all sets using training statistics
    X_train_norm = (X_train - train_mean) / train_std
    
    results = [X_train_norm]
    
    if X_val is not None:
        X_val_norm = (X_val - train_mean) / train_std
        results.append(X_val_norm)
    
    if X_test is not None:
        X_test_norm = (X_test - train_mean) / train_std
        results.append(X_test_norm)
    
    # Return statistics as well (useful for inverse transform if needed)
    results.append((train_mean, train_std))
    
    return tuple(results)


if __name__ == "__main__":
    # Demo: Load Apple stock data
    print("Demo: Loading AAPL data")
    
    loader = PriceDataLoader(
        ticker="AAPL",
        start_date="2015-01-01",
        end_date="2023-12-31",
        window_size=20
    )
    
    X, y, dates = loader.load_and_prepare()
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"{'='*60}")
    print(f"You now have:")
    print(f"  - X: {X.shape} (price windows)")
    print(f"  - y: {y.shape} (return labels)")
    print(f"  - dates: {len(dates)} timestamps")
