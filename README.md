# Stock Return Prediction - Price-Based Baseline

This project implements a **price-based baseline model** for predicting next-day stock returns.

## 🎯 Goal

**Input:** Past W days of daily closing prices  
**Output:** Next-day expected return  

Return definition: `return_{t+1} = (P_{t+1} - P_t) / P_t`

This is a **regression task**.

---

## 📁 Project Structure

```
capstone/
│
├── README.md                          # This file
│
├── config/
│   └── price_baseline.yaml            # Hyperparameters
│
├── data/
│   ├── raw/                          # Downloaded price data
│   ├── processed/                    # Preprocessed data (optional)
│   └── README.md                     # Data documentation
│
├── src/
│   ├── data/
│   │   ├── price_loader.py           # Data loading and windowing
│   │   └── splits.py                 # Train/val/test splitting
│   │
│   ├── models/
│   │   └── price_model.py            # LSTM and MLP models
│   │
│   ├── training/
│   │   └── train_price_model.py      # Training script
│   │
│   ├── evaluation/
│   │   └── metrics.py                # Evaluation metrics
│   │
│   └── utils/
│       └── seed.py                   # Reproducibility
│
├── experiments/
│   └── price_only_baseline.ipynb     # Demo notebook
│
├── results/
│   └── price_only/
│       ├── metrics.txt               # Test set metrics
│       ├── loss_curve.png            # Training curves
│       └── checkpoints/              # Saved models
│
└── notes/
    └── DAY_1_HACKATHON_REPORT.md     # Development notes
```

---

## 🚀 Quick Start

### Option 1: Interactive Multi-Stock Dashboard (Recommended ⭐)

Launch the interactive Streamlit dashboard to train and visualize any stock:

```bash
streamlit run streamlit_app.py
```

**Features:**
- 🎯 Train models for 25+ popular stocks or any custom ticker
- 📊 Real-time training progress with epoch updates
- 📈 Dark theme visualizations (Apple Stocks inspired)
- 💾 Save and load trained models automatically
- 📥 Export predictions to CSV

See [STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md) for detailed usage instructions.

### Option 2: One-Command Training (AAPL)

Quick baseline training for Apple stock:

```bash
python quick_start.py
```

Results saved to `results/price_only/`.

### Option 3: Command-Line Training

```bash
python -m src.training.train_price_model --config config/price_baseline.yaml
```

### Option 4: Visualize Trained Model

Generate static visualizations from trained AAPL model:

```bash
python visualize_results.py
```

Outputs:
- `results/price_only/AAPL_prediction_visualization.png` - Clean view
- `results/price_only/AAPL_detailed_analysis.png` - 4-panel analysis

---

## 📦 Installation

### 1. Install Core Dependencies

```bash
pip install torch numpy pandas yfinance matplotlib pyyaml
```

### 2. Install Streamlit (for dashboard)

```bash
pip install streamlit
```

Or install all at once:

```bash
pip install torch numpy pandas yfinance matplotlib pyyaml streamlit
```

---

## 📊 What This Baseline Does

### Data Pipeline
1. **Download** historical prices from Yahoo Finance (yfinance)
2. **Compute** next-day returns: `(P_{t+1} - P_t) / P_t`
3. **Create** sliding windows of size W (default: 20 days)
4. **Split** chronologically: 70% train / 15% val / 15% test
5. **Normalize** using training set statistics only

### Model
- **Architecture:** LSTM (2 layers, 64 hidden units)
- **Input:** Window of past W normalized prices
- **Output:** Single scalar (predicted return)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam

### Evaluation Metrics
- **MSE:** Mean Squared Error
- **MAE:** Mean Absolute Error  
- **RMSE:** Root Mean Squared Error
- **Directional Accuracy:** Percent of correct sign predictions
  - This is what matters for trading!
  - > 50% means better than random

---

## 🔧 Configuration

Edit `config/price_baseline.yaml` to change:
- Ticker symbol
- Date range
- Window size
- Model hyperparameters
- Training settings

Example:
```yaml
data:
  ticker: "AAPL"
  start_date: "2015-01-01"
  end_date: "2023-12-31"
  window_size: 20

model:
  type: "lstm"
  params:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.2

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

---

## 📝 Key Design Principles

### ✅ What This Code Does RIGHT
1. **No future information leakage**
   - Time-based splits (not random)
   - Normalization uses training stats only
   - Windows use only past data

2. **Clean and documented**
   - Every function has clear docstrings
   - Extensive logging and debug prints
   - Easy to understand and modify

3. **Proper evaluation**
   - Financial metrics (directional accuracy)
   - Test set is never seen during training
   - Reports metrics on all splits

### ⚠️ What This Code Does NOT Do (Yet)
- News data (future extension)
- Graph neural networks (future extension)
- Federated learning (future extension)
- Portfolio optimization
- Risk management

**This is intentional.** We build a solid baseline first, then add complexity.

---

## 🧪 Experiments

See `experiments/price_only_baseline.ipynb` for:
- Interactive data exploration
- Model training demo
- Results visualization
- Prediction examples

---

## 📈 Expected Performance

**Realistic expectations for price-only models:**
- Directional accuracy: 50-55% (barely better than random)
- Returns are very noisy and hard to predict
- This is why we need multimodal data (news, graphs, etc.)

**What makes a good baseline:**
- Correct implementation (no bugs)
- Proper evaluation protocol
- Reproducible results
- Room for extension

---

## 🔮 Next Steps

This baseline will be extended with:

1. **News Data Integration**
   - Scrape financial news
   - Use sentiment analysis or embeddings
   - Combine with price data

2. **Graph Transformer**
   - Model stock relationships
   - Use graph neural networks
   - Capture market structure

3. **Federated Learning**
   - Train on distributed data
   - Privacy-preserving
   - Scalable

But first, we make sure the baseline is solid! 🎯

---

## 📚 References

- Stock data: [Yahoo Finance](https://finance.yahoo.com/)
- `yfinance` library: [GitHub](https://github.com/ranaroussi/yfinance)
- PyTorch: [pytorch.org](https://pytorch.org/)

---

## 👤 Author

Capstone Project - Stock Return Prediction  
Date: February 2026

---

## 📄 License

Educational project - use freely for learning!
