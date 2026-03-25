"""
data_loader.py
==============
Loads and preprocesses the StockNet dataset for the baseline LR pipeline.

Responsibilities
----------------
- Load the CSV, parse dates, sort by (Ticker, Date)
- Fix Volume_Change infinities
- Lag all same-day technical features by 1 trading day per ticker
  (prevents target leakage — Return is literally sign(Return) == Target)
- Forward-fill fundamentals within each ticker; add binary presence masks
- Define the three chronological splits
- Z-score normalise using TRAIN statistics only

Usage
-----
    from data_loader import load_data, get_feature_groups, SPLITS

    df = load_data("stocknet_final_modeling_set.csv")
    feat = get_feature_groups()
    X_train, y_train = df[SPLITS["train"]][feat["price"]], df[SPLITS["train"]]["Target"]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ── Column definitions ─────────────────────────────────────────────────────────

# Same-day technical indicators — must be lagged to avoid leakage
_TECHNICAL_RAW = [
    "Return", "MA_5", "MA_10", "MA_20",
    "Price_MA5_Ratio", "Price_MA10_Ratio", "Price_MA20_Ratio",
    "Volatility_5", "Volatility_20",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "Volume_Change", "HL_Spread",
]

# Tweet counts (same-day — must also be lagged)
_SENTIMENT_RAW = [
    "Company_Tweet_Count", "Event_Tweet_Count", "Total_Tweet_Count",
]

# All 14 EDGAR fundamental columns
FUNDAMENTAL_COLS = [
    "Revenue", "NetIncome", "TotalAssets", "TotalLiabilities",
    "StockholdersEquity", "OperatingIncome", "EPS", "Cash",
    "Profit_Margin", "Debt_To_Equity", "ROA",
    "Current_Ratio", "Asset_Turnover", "Operating_Margin",
]

# Date boundaries for chronological split
SPLITS = {
    "train": ("2014-01-01", "2015-03-31"),
    "val":   ("2015-04-01", "2015-07-31"),
    "test":  ("2015-08-01", "2015-12-31"),
}

TARGET = "Target"


# ── Public API ─────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the StockNet CSV, apply all preprocessing steps, and return a
    clean DataFrame ready for feature extraction.

    Parameters
    ----------
    csv_path : str
        Path to stocknet_final_modeling_set.csv

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with lagged features and fundamental masks added.
        Rows where lag-1 features are NaN (first row per ticker) are dropped.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[data_loader] Loading {path.name} ...")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"[data_loader] Raw shape: {df.shape}")

    df = _fix_infinities(df)
    df = _lag_same_day_features(df)
    df = _process_fundamentals(df)

    # Drop rows where any lag-1 feature is NaN (first row per ticker)
    lag_cols = [f"{c}_lag1" for c in _TECHNICAL_RAW + _SENTIMENT_RAW]
    before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    print(f"[data_loader] Dropped {before - len(df)} rows (lag NaN). Final shape: {df.shape}")

    _print_split_sizes(df)
    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Return a dict of named feature-column lists.

    Keys
    ----
    price          : 15 lagged technical indicators
    sentiment      : 3 lagged tweet-count features
    fundamentals   : 14 EDGAR values + 14 binary presence masks (28 total)
    price_sentiment: price + sentiment
    price_fund     : price + fundamentals
    full           : price + sentiment + fundamentals
    """
    price_feats  = [f"{c}_lag1" for c in _TECHNICAL_RAW]
    sent_feats   = [f"{c}_lag1" for c in _SENTIMENT_RAW]
    fund_feats   = FUNDAMENTAL_COLS + [f"{c}_present" for c in FUNDAMENTAL_COLS]

    return {
        "price":            price_feats,
        "sentiment":        sent_feats,
        "fundamentals":     fund_feats,
        "price_sentiment":  price_feats + sent_feats,
        "price_fund":       price_feats + fund_feats,
        "full":             price_feats + sent_feats + fund_feats,
    }


def get_split_mask(df: pd.DataFrame, split: str) -> pd.Series:
    """
    Return a boolean mask for the requested split.

    Parameters
    ----------
    split : "train" | "val" | "test"
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS.keys())}, got '{split}'")
    start, end = SPLITS[split]
    return (df["Date"] >= start) & (df["Date"] <= end)


def get_xy(
    df: pd.DataFrame,
    features: list,
    split: str,
    scaler: StandardScaler = None,
    fit_scaler: bool = False,
):
    """
    Extract (X, y) arrays for a given split, optionally fitting/applying a scaler.

    Parameters
    ----------
    df         : preprocessed DataFrame from load_data()
    features   : list of column names
    split      : "train" | "val" | "test"
    scaler     : an sklearn StandardScaler (or None to create one)
    fit_scaler : if True, fit the scaler on this split's data (use for train only)

    Returns
    -------
    X          : np.ndarray of shape (N, F)
    y          : np.ndarray of shape (N,)
    scaler     : the scaler (fitted if fit_scaler=True, else passed-through)
    """
    mask = get_split_mask(df, split)
    X = df.loc[mask, features].fillna(0).values.astype(np.float32)
    y = df.loc[mask, TARGET].values

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fix_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf in Volume_Change with clipped values; fill residual NaN."""
    df = df.copy()
    df["Volume_Change"] = df["Volume_Change"].replace([np.inf, -np.inf], np.nan)
    low  = df["Volume_Change"].quantile(0.01)
    high = df["Volume_Change"].quantile(0.99)
    df["Volume_Change"] = df["Volume_Change"].clip(lower=low, upper=high)
    df["Volume_Change"]  = df["Volume_Change"].fillna(df["Volume_Change"].median())
    df["Volatility_20"]  = df["Volatility_20"].fillna(df["Volatility_20"].median())
    print(f"[data_loader] Fixed Volume_Change infinities (clipped to [{low:.4f}, {high:.4f}])")
    return df


def _lag_same_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag-1 versions of all same-day technical and sentiment features.

    WHY: Return = (Close_t - Close_{t-1}) / Close_{t-1}
         Target = 1 if Return > 0, else 0
         => Using Return directly gives 100% accuracy (leakage).
         All technical indicators are also computed from today's Close.
         We shift each feature by 1 row within each ticker group so the
         model only sees YESTERDAY's signal when predicting TODAY's Target.
    """
    df = df.copy()
    cols_to_lag = _TECHNICAL_RAW + _SENTIMENT_RAW
    for col in cols_to_lag:
        df[f"{col}_lag1"] = df.groupby("Ticker")[col].shift(1)
    print(f"[data_loader] Created lag-1 versions of {len(cols_to_lag)} columns")
    return df


def _process_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle ~35-50% missing EDGAR fundamentals.

    Strategy
    --------
    1. Forward-fill within each ticker (filing data is valid until the next filing)
    2. Add binary presence mask columns (1 = value known, 0 = still missing)
    3. Zero-fill any remaining NaN (cross-ticker gaps at start of history)
    """
    df = df.copy()
    for col in FUNDAMENTAL_COLS:
        df[col] = df.groupby("Ticker")[col].transform(lambda x: x.ffill())
        df[f"{col}_present"] = (~df[col].isna()).astype(int)
    df[FUNDAMENTAL_COLS] = df[FUNDAMENTAL_COLS].fillna(0)
    print(f"[data_loader] Processed {len(FUNDAMENTAL_COLS)} fundamental columns with ffill + mask")
    return df


def _print_split_sizes(df: pd.DataFrame) -> None:
    for split, (start, end) in SPLITS.items():
        mask = (df["Date"] >= start) & (df["Date"] <= end)
        n    = mask.sum()
        pos  = df.loc[mask, TARGET].mean()
        print(f"[data_loader]   {split:5s}: {n:5d} rows | {start} to {end} | "
              f"class balance: {pos*100:.1f}% up")
