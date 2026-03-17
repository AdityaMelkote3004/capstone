# Multi-Stock Prediction Dashboard - User Guide

## 🚀 Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## 🎯 Features

### 1. Stock Selection

**Choose from 25+ Popular Stocks** across 5 categories:
- **Technology:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- **Finance:** JPM, BAC, WFC, GS, MS
- **Healthcare:** JNJ, PFE, UNH, CVS, ABBV
- **Consumer:** WMT, HD, MCD, NKE, SBUX
- **Energy:** XOM, CVX, COP, SLB, OXY

**Custom Ticker Input:**
- Check "Use custom ticker" to enter any valid stock symbol
- Works with any ticker available on Yahoo Finance

### 2. Configuration Options

**Date Range:**
- Start Date: Historical data beginning (default: 2015-01-01)
- End Date: Historical data end (default: 2023-12-31)

**Window Size:**
- Number of past days to use for prediction (5-60)
- Default: 20 days
- Larger windows = more historical context, slower training

**Advanced Settings (expandable):**
- Training Epochs: 10-100 (default: 50)
- Batch Size: 16-128 (default: 32)
- Learning Rate: 0.0001-0.01 (default: 0.001)
- LSTM Hidden Dimension: 32-128 (default: 64)
- LSTM Layers: 1-4 (default: 2)
- Dropout: 0.0-0.5 (default: 0.2)

### 3. Training Workflow

**First-Time Training:**
1. Select a stock ticker
2. Adjust configuration if needed
3. Click "🚀 Train New Model"
4. Watch real-time progress bar with epoch updates
5. Wait for training to complete (1-5 minutes)

**Retraining:**
1. If a trained model exists, you'll see "✅ Trained model found"
2. Click "🔄 Retrain Model" to train with new parameters
3. Previous model will be overwritten

**Model Storage:**
- Models saved to `results/multi_stock/{TICKER}/`
- Each ticker has its own checkpoint
- Automatically loads pre-trained models on selection

---

## 📊 Understanding the Dashboard

### Data Summary (Top Row)

After loading data, you'll see:
- **Train Samples:** Number of training examples
- **Val Samples:** Number of validation examples  
- **Test Samples:** Number of test examples
- **Window Size:** Configured lookback period

### Performance Metrics

**MSE (Mean Squared Error):**
- Lower is better
- Measures average squared prediction error
- Typical range: 0.0001 - 0.001

**MAE (Mean Absolute Error):**
- Displayed as percentage
- Average absolute prediction error
- Good performance: < 2%

**RMSE (Root Mean Squared Error):**
- Displayed as percentage
- Square root of MSE, more interpretable
- Good performance: < 2%

**Directional Accuracy:**
- Most important metric for trading
- Percentage of correctly predicted up/down movements
- **> 55%:** Significantly better than random ✅
- **50-55%:** Slightly better than random 📊
- **< 50%:** Below random, needs improvement ⚠️

### Visualizations

**1. Time Series Analysis (Main Chart)**
- Green line: Actual returns
- Blue dashed line: Predicted returns
- Gray horizontal line: Zero return baseline
- Shows model tracking ability over time

**2. Scatter Plot**
- Each point: one prediction
- Red dashed line: Perfect prediction
- Points closer to line = better predictions
- Shows prediction accuracy distribution

**3. Error Distribution**
- Histogram of prediction errors
- Green dashed line: Zero error
- Centered around zero = unbiased model
- Narrower distribution = more accurate

**4. Training History** (if model just trained)
- Green: Training loss over epochs
- Blue: Validation loss over epochs
- Decreasing trend = learning progress
- Gap between lines = overfitting indicator

### Raw Predictions Table (Expandable)

View detailed predictions with:
- Date
- Actual return
- Predicted return
- Error (Predicted - Actual)
- Correct Direction (boolean)

**Download CSV:**
- Export all predictions for external analysis
- File format: `{TICKER}_predictions.csv`

---

## 💡 Usage Tips

### For Quick Testing
1. Select AAPL (most reliable data)
2. Use default settings
3. Train with 20-30 epochs for speed

### For Best Performance
1. Use full date range (2015-2023)
2. Train for 50+ epochs
3. Window size 20-30 days optimal for most stocks
4. Learning rate 0.001 works well for most cases

### For Experimentation
1. Try different window sizes (10, 20, 40)
2. Adjust hidden dimensions (32, 64, 128)
3. Compare results across different stocks
4. Test volatile stocks (TSLA, NVDA) vs stable (JPM, WMT)

### Performance Expectations
- **Price-only models typically achieve 45-52% directional accuracy**
- Below random performance is expected - validates need for multimodal approach
- Higher volatility stocks = harder to predict
- Longer training periods = more stable results

---

## 🎨 Visual Theme

Dashboard uses **dark theme inspired by Apple Stocks app:**
- Black background (`#000000`)
- Green for positive/actual values (`#30D158`)
- Blue for predictions (`#0A84FF`)
- Red for errors/negative (`#FF453A`)
- Gray accents (`#48484A`, `#1C1C1E`)

---

## 🔧 Troubleshooting

### "Failed to load data"
- Check ticker symbol is valid
- Verify date range has sufficient data
- Try expanding date range (need >window_size days)

### Training is slow
- Reduce number of epochs
- Reduce batch size
- Reduce hidden dimensions
- Close other applications using GPU

### Model not loading
- Delete cached model and retrain
- Check `results/multi_stock/{TICKER}/` exists
- Verify checkpoint file not corrupted

### Memory errors
- Reduce batch size
- Reduce hidden dimensions
- Reduce date range
- Close other applications

---

## 📈 Multi-Stock Workflow

### Comparing Multiple Stocks

**Quick Comparison:**
1. Train model for Stock A
2. Note directional accuracy
3. Switch to Stock B in sidebar
4. Train model for Stock B
5. Compare metrics manually

**Manual Analysis:**
1. Download CSV predictions for each stock
2. Import into Excel/Python
3. Create comparison charts
4. Analyze sector-specific patterns

### Best Stocks for Baseline Testing

**Technology (High Volatility):**
- AAPL, MSFT - Most reliable
- NVDA, TSLA - Most challenging

**Finance (Medium Volatility):**
- JPM, BAC - Consistent patterns
- GS, MS - More volatile

**Healthcare (Low Volatility):**
- JNJ - Very stable
- UNH - Growing sector

**Consumer (Defensive):**
- WMT - Low volatility
- NKE - Brand strength

**Energy (Commodity-Linked):**
- XOM, CVX - Oil price correlation

---

## 🎓 Understanding Results

### What Good Looks Like

**Successful Model:**
- Directional accuracy > 50%
- Validation loss decreasing steadily
- Predictions follow actual trend
- Error distribution centered at zero

**Struggling Model:**
- Directional accuracy < 50%
- Validation loss plateaus early
- Predictions lag actual movements
- High error variance

### Why Price-Only Struggles

**Expected Behavior:**
- Directional accuracy 45-52% (around random)
- Difficulty capturing trend changes
- High sensitivity to noise

**This is intentional:**
- Validates baseline purpose
- Proves need for additional features
- Sets foundation for multimodal extensions

### Next Steps After Baseline

1. **Add Technical Indicators:** RSI, MACD, Bollinger Bands
2. **Incorporate Volume Data:** Trading volume patterns
3. **Include Market Sentiment:** News, social media
4. **Multi-Stock Correlation:** Market-wide features
5. **Ensemble Methods:** Combine multiple models

---

## 🔗 Related Documentation

- [README.md](README.md) - Project overview
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Static visualization details
- [DAY_1_HACKATHON_REPORT.md](notes/DAY_1_HACKATHON_REPORT.md) - Development notes
- [price_baseline.yaml](config/price_baseline.yaml) - Configuration reference

---

## 🎯 Quick Reference

**Key Shortcuts:**
- `Ctrl+C` in terminal - Stop Streamlit server
- `R` in browser - Rerun app
- `C` in browser - Clear cache
- `A` in browser - Toggle theme (light/dark)

**Important Paths:**
- Models: `results/multi_stock/{TICKER}/checkpoints/best_model.pt`
- Exports: `{TICKER}_predictions.csv` (downloaded to browser)

**Default Settings:**
- Window: 20 days
- Epochs: 50
- Batch: 32
- LR: 0.001
- Hidden: 64
- Layers: 2
- Dropout: 0.2

---

Built with ❤️ using Streamlit & PyTorch
