"""StockNet Dataset with per-ticker 80/20 time-based split and 4 feature sets."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict

# ── Feature set definitions ────────────────────────────────

# FS1: Price Only — 14 technical indicators
PRICE_FEATURES = [
    'Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'Volatility_5', 'Volatility_20',
    'Price_MA5_Ratio', 'Price_MA10_Ratio', 'Price_MA20_Ratio',
    'Volume_Change', 'HL_Spread', 'MA_5', 'MA_10',
]

# FS2 addition: 8 EDGAR fundamental columns
FUNDAMENTAL_FEATURES = [
    'Revenue', 'NetIncome', 'TotalAssets', 'TotalLiabilities',
    'StockholdersEquity', 'EPS', 'Cash', 'ROA',
]

# FS3 addition: 3 tweet count / sentiment columns
TWEET_FEATURES = [
    'Company_Tweet_Count', 'Event_Tweet_Count', 'Total_Tweet_Count',
]

# All fundamentals (used for cleaning)
ALL_FUNDAMENTAL_COLS = [
    'Revenue', 'NetIncome', 'TotalAssets', 'TotalLiabilities',
    'StockholdersEquity', 'EPS', 'Cash', 'ROA',
    'Profit_Margin', 'Debt_To_Equity', 'Current_Ratio',
    'Asset_Turnover', 'Operating_Margin',
]

FEATURE_SETS: Dict[str, List[str]] = {
    'FS1_Price':              PRICE_FEATURES,
    'FS2_Price_Fundamentals': PRICE_FEATURES + FUNDAMENTAL_FEATURES,
    'FS3_Price_Tweets':       PRICE_FEATURES + TWEET_FEATURES,
    'FS4_Full_Structured':    PRICE_FEATURES + FUNDAMENTAL_FEATURES + TWEET_FEATURES,
}

SECTOR_MAP = {
    'Basic_Materials': 0, 'Conglomerates': 1, 'Consumer_Goods': 2,
    'Financial': 3, 'Healthcare': 4, 'Industrial_Goods': 5,
    'Services': 6, 'Technology': 7, 'Utilities': 8,
}


# ── Data loading & cleaning ────────────────────────────────

def load_and_clean(parquet_path: str) -> pd.DataFrame:
    """Load parquet and fix known data quality issues."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Fix infinities in Volume_Change
    df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], 0.0)

    # Fill remaining nulls in price/technical features
    for col in PRICE_FEATURES + TWEET_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(0.0)

    # Forward-fill fundamentals within each ticker, then zero-fill
    for col in ALL_FUNDAMENTAL_COLS:
        if col in df.columns:
            df[col] = df.groupby('Ticker')[col].ffill()
            df[col] = df[col].fillna(0.0)

    df['Sector_ID'] = df['Sector'].map(SECTOR_MAP).fillna(0).astype(int)

    return df


# ── Per-ticker 80/20 time-based split ─────────────────────

def split_per_ticker(df: pd.DataFrame, train_ratio: float = 0.8
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-ticker chronological split as per the flowchart:
    For each ticker, sort by Date, take first 80% as train, last 20% as test.
    Aggregate all tickers' train/test sets.
    """
    train_parts, test_parts = [], []

    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date').reset_index(drop=True)
        n = len(group)
        cutoff = int(n * train_ratio)
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    return train_df, test_df


def compute_norm_stats(train_df: pd.DataFrame,
                       feature_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    means = train_df[feature_cols].mean()
    stds  = train_df[feature_cols].std().replace(0, 1.0)
    return means, stds


def normalize(df: pd.DataFrame, feature_cols: List[str],
              means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - means) / stds
    return df


# ── Dataset class ──────────────────────────────────────────

class StockNetDataset(Dataset):
    """
    PyTorch Dataset for MMGTFFF.

    Builds sliding windows of W days per ticker.
    feature_set selects which columns to use:
        'FS1_Price', 'FS2_Price_Fundamentals',
        'FS3_Price_Tweets', 'FS4_Full_Structured'
    """

    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 window_size: int = 5):
        self.feature_cols = feature_cols
        self.window_size  = window_size
        self.samples      = []
        self._build(df)

    def _build(self, df: pd.DataFrame):
        for ticker, group in df.groupby('Ticker'):
            group  = group.sort_values('Date').reset_index(drop=True)
            feats  = group[self.feature_cols].values.astype(np.float32)
            labels = group['Target'].values.astype(np.int64)
            dates  = group['Date'].values
            sids   = group['Sector_ID'].values.astype(np.int64)
            ctexts = group['Company_Texts'].values
            etexts = group['Event_Texts'].values

            for i in range(self.window_size, len(group)):
                self.samples.append({
                    'window':    feats[i - self.window_size:i],  # (W, F) days [i-W .. i-1]
                    'flat':      feats[i - 1],                    # (F,) yesterday — no leakage
                    'target':    labels[i],                       # today's direction
                    'ticker':    ticker,
                    'date':      dates[i],
                    'sector_id': sids[i],
                    'company_text': ctexts[i],
                    'event_text':   etexts[i],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'window':    torch.tensor(s['window'], dtype=torch.float32),
            'flat':      torch.tensor(s['flat'],   dtype=torch.float32),
            'target':    torch.tensor(s['target'], dtype=torch.long),
            'sector_id': torch.tensor(s['sector_id'], dtype=torch.long),
        }

    def get_metadata(self, idx) -> dict:
        s = self.samples[idx]
        return {k: s[k] for k in ('ticker', 'date', 'company_text', 'event_text')}

    def to_numpy(self, use_window: bool = False):
        """Return (X, y) numpy arrays for sklearn models."""
        X = np.array([
            s['window'].flatten() if use_window else s['flat']
            for s in self.samples
        ])
        y = np.array([s['target'] for s in self.samples])
        return X, y


# ── Main entry point ───────────────────────────────────────

def build_datasets(parquet_path: str, feature_set: str = 'FS1_Price',
                   window_size: int = 5, train_ratio: float = 0.8,
                   ) -> Tuple[StockNetDataset, StockNetDataset, dict]:
    """
    Full pipeline:
      load → clean → per-ticker 80/20 split → normalize → build datasets.

    Returns (train_dataset, test_dataset, info_dict).
    Note: no separate val set per the flowchart; training scripts use
    a held-out 10% of train for early stopping.
    """
    assert feature_set in FEATURE_SETS, \
        f"feature_set must be one of {list(FEATURE_SETS.keys())}"

    feature_cols = FEATURE_SETS[feature_set]
    df = load_and_clean(parquet_path)

    train_df, test_df = split_per_ticker(df, train_ratio)

    # Normalize using train stats only
    means, stds = compute_norm_stats(train_df, feature_cols)
    train_df = normalize(train_df, feature_cols, means, stds)
    test_df  = normalize(test_df,  feature_cols, means, stds)

    train_ds = StockNetDataset(train_df, feature_cols, window_size)
    test_ds  = StockNetDataset(test_df,  feature_cols, window_size)

    # Per-ticker split stats
    ticker_splits = {}
    for ticker, group in df.groupby('Ticker'):
        n = len(group)
        ticker_splits[ticker] = {
            'total': n,
            'train': int(n * train_ratio),
            'test':  n - int(n * train_ratio),
        }

    info = {
        'feature_set':     feature_set,
        'feature_cols':    feature_cols,
        'num_features':    len(feature_cols),
        'window_size':     window_size,
        'num_tickers':     df['Ticker'].nunique(),
        'tickers':         sorted(df['Ticker'].unique().tolist()),
        'sectors':         sorted(df['Sector'].unique().tolist()),
        'train_size':      len(train_ds),
        'test_size':       len(test_ds),
        'train_rows':      len(train_df),
        'test_rows':       len(test_df),
        'train_pct':       round(len(train_df) / len(df) * 100, 1),
        'test_pct':        round(len(test_df)  / len(df) * 100, 1),
        'norm_means':      means.to_dict(),
        'norm_stds':       stds.to_dict(),
        'ticker_splits':   ticker_splits,
    }
    return train_ds, test_ds, info
