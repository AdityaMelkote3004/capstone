# Data Directory

This directory stores all data for the stock return prediction project.

## Structure

```
data/
├── raw/
│   ├── prices/          # Raw price data from Yahoo Finance
│   │   ├── AAPL.csv
│   │   ├── MSFT.csv
│   │   └── ...
│   └── news/           # Raw news data (future)
│       └── financial_news.csv
│
└── processed/          # Preprocessed data (optional)
    ├── price_windows.npy
    └── price_labels.npy
```

## Data Sources

### Price Data
- **Source:** Yahoo Finance (via yfinance)
- **Frequency:** Daily
- **Fields:** Open, High, Low, Close, Volume, Adj Close
- **Used:** Close price only (for baseline)

### Download Data
Data is automatically downloaded when you run the training script.

Alternatively, download manually:
```python
from src.data.price_loader import PriceDataLoader

loader = PriceDataLoader(
    ticker="AAPL",
    start_date="2015-01-01",
    end_date="2023-12-31",
    window_size=20
)
X, y, dates = loader.load_and_prepare()
```

## Data Format

### Raw Prices
```
Date,Close
2015-01-02,109.33
2015-01-05,106.25
2015-01-06,106.26
...
```

### Processed Windows
- **X:** (num_samples, window_size, 1)
  - Window of past prices
- **y:** (num_samples,)
  - Next-day return labels
- **dates:** (num_samples,)
  - Timestamp for each sample

## Important Notes

### No Future Information Leakage
- Data is split chronologically
- Training data comes before test data
- Normalization uses training statistics only

### Missing Values
- Forward filled if present
- Uncommon in Yahoo Finance data
- Logged if found

### Returns Calculation
```
return_t = (price_t - price_{t-1}) / price_{t-1}
```

## Data Requirements

For the baseline model:
- Minimum: ~500 days of price data
- Recommended: 2000+ days (for meaningful train/val/test)
- Window size: 20 days (default)

With 2000 days of data:
- Train: ~1400 samples
- Val: ~300 samples
- Test: ~300 samples

## Adding New Tickers

To train on a different stock, edit `config/price_baseline.yaml`:
```yaml
data:
  ticker: "MSFT"  # Change this
```

Or create a custom config file.

## Future Extensions

This directory will later include:
- News data (scraped from financial sources)
- Sentiment scores
- Technical indicators
- Market graphs

For now, we focus on prices only! 📈
