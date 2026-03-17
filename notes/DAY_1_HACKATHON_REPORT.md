# Day 1 Hackathon Report: Price-Based Baseline Model

**Date:** February 2, 2026  
**Project:** Stock Return Prediction  
**Phase:** Baseline Implementation (Price Data Only)

---

## 🎯 Objective

Build a **clean, correct, and extensible baseline** for stock return prediction using only historical closing prices.

**No news. No graphs. No federated learning. Just prices.**

---

## ✅ What Was Accomplished

### 1. Complete Project Structure

Created a well-organized codebase:

```
capstone/
├── config/          # Hyperparameter configurations
├── data/            # Raw and processed data
├── src/             # Source code (modular)
│   ├── data/        # Data loading and splitting
│   ├── models/      # Model architectures
│   ├── training/    # Training scripts
│   ├── evaluation/  # Metrics computation
│   └── utils/       # Utilities (seed, etc.)
├── experiments/     # Jupyter notebooks
├── results/         # Saved models and metrics
└── notes/           # Documentation
```

### 2. Data Pipeline (src/data/)

**price_loader.py:**
- Downloads historical data from Yahoo Finance (yfinance)
- Computes next-day returns: `return_{t+1} = (P_{t+1} - P_t) / P_t`
- Creates sliding windows: input shape `(num_samples, W, 1)`
- Handles missing values appropriately
- NO future information leakage

**splits.py:**
- Time-based train/val/test splitting (NOT random)
- Ensures train < val < test chronologically
- Also supports explicit date-based splits
- Extensive logging for debugging

### 3. Model Architecture (src/models/)

**price_model.py:**
- Implemented LSTM baseline (2 layers, 64 hidden units)
- Input: sequence of past W prices (normalized)
- Output: single scalar (predicted return)
- Also includes MLP alternative for comparison

**Design choices:**
- Regression task (not classification)
- No softmax/sigmoid
- Simple and interpretable

### 4. Training Loop (src/training/)

**train_price_model.py:**
- Standard supervised learning loop
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Tracks train and validation losses
- Saves best model based on validation loss
- Generates loss curves and metrics

### 5. Evaluation Metrics (src/evaluation/)

**metrics.py:**
- **MSE:** Standard regression metric
- **MAE:** Robust to outliers
- **RMSE:** Same units as returns
- **Directional Accuracy:** Percent of correct sign predictions
  - **This is what matters for trading!**

### 6. Configuration System

**config/price_baseline.yaml:**
- All hyperparameters in one place
- Easy to modify and experiment
- Version control friendly

### 7. Demo Notebook

**experiments/price_only_baseline.ipynb:**
- Complete end-to-end demo
- Data visualization
- Model training
- Results analysis
- Ready for presentation

### 8. Documentation

**README.md:**
- Clear project overview
- Quick start guide
- Explanation of design choices
- Next steps outlined

**data/README.md:**
- Data structure documentation
- Download instructions
- Format specifications

---

## 📊 Technical Details

### Problem Formulation

- **Input:** Past W days of closing prices (W=20 default)
- **Output:** Next-day expected return
- **Task:** Regression
- **Evaluation:** MSE, MAE, Directional Accuracy

### Data Specifications

- **Source:** Yahoo Finance via yfinance
- **Ticker:** AAPL (configurable)
- **Period:** 2015-2023 (configurable)
- **Frequency:** Daily
- **Features:** Close price only

### Model Architecture

```
Input: (batch, 20, 1)
  ↓
LSTM (2 layers, hidden=64)
  ↓
FC Layer (64 → 1)
  ↓
Output: (batch, 1)  [predicted return]
```

### Training Configuration

- **Epochs:** 50
- **Batch size:** 32
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Loss:** MSE
- **Splits:** 70% train / 15% val / 15% test

---

## 🔬 Engineering Principles Applied

### ✅ Correctness First
- Time-based splits (no leakage)
- Normalization using training stats only
- Proper validation protocol
- Extensive debugging prints

### ✅ Clarity and Readability
- Every function has clear docstrings
- Explains what, how, and why
- Comments explain intent, not just mechanics
- Easy for others to understand

### ✅ Modularity
- Separate files for data, models, training, evaluation
- Easy to swap components
- Testable in isolation
- Reusable for future extensions

### ✅ Reproducibility
- Random seed setting
- Deterministic operations
- Configuration versioning
- Results tracking

### ✅ Extensibility
- Clean interfaces for adding features
- Ready for multimodal data
- Ready for graph integration
- Ready for federated learning

---

## 📈 Expected Performance

### Realistic Baselines

For price-only models on stock returns:
- **Directional Accuracy:** 50-55%
  - Barely better than random (50%)
  - This is expected! Returns are very noisy
- **MSE/MAE:** Small values (returns are typically < 0.05)

### Why This is Good

1. **Correct implementation** (no bugs)
2. **Proper evaluation** (no cheating)
3. **Room for improvement** (clear path forward)

**The point of a baseline is NOT to be state-of-the-art.**  
**The point is to be CORRECT and EXTENSIBLE.**

---

## 🚀 Next Steps (Future Phases)

### Phase 2: News Integration
- Scrape financial news
- Sentiment analysis or embeddings
- Multimodal fusion (prices + news)
- Expected improvement: +2-5% directional accuracy

### Phase 3: Graph Transformer
- Model stock relationships
- Graph neural networks
- Market structure modeling
- Expected improvement: +3-7% directional accuracy

### Phase 4: Federated Learning
- Distributed training
- Privacy preservation
- Scalability
- Production readiness

---

## 💡 Key Learnings

### What Worked Well
1. **Modular design** makes debugging easy
2. **Time-based splits** prevent leakage
3. **Extensive logging** helps understand behavior
4. **Simple baseline** is easy to explain and extend

### Challenges Encountered
1. **Data quality:** Some tickers have missing values
   - Solution: Forward fill with logging
2. **Hyperparameter tuning:** Finding good learning rate
   - Solution: Start with standard values (0.001)
3. **Overfitting:** LSTM can easily overfit
   - Solution: Dropout and early stopping

### Surprising Insights
1. **Price-only models are weak** (as expected)
   - Reinforces need for multimodal approach
2. **Directional accuracy is hard**
   - Even small improvements are valuable
3. **Simple LSTM works** (no need for complex architectures yet)

---

## 📦 Deliverables

### Code
- ✅ Modular, documented Python codebase
- ✅ Configuration system
- ✅ Training script
- ✅ Evaluation metrics
- ✅ Demo notebook

### Documentation
- ✅ README with quick start
- ✅ Code comments and docstrings
- ✅ This hackathon report

### Results
- ✅ Trained model checkpoints
- ✅ Evaluation metrics
- ✅ Loss curves
- ✅ Prediction visualizations

---

## 🎓 Conclusion

**Mission accomplished!** ✨

We now have:
1. **Clean baseline** that works
2. **Proper evaluation** protocol
3. **Extensible architecture** for future phases
4. **Clear path forward** for improvements

The baseline is intentionally simple. The goal was NOT to achieve state-of-the-art performance, but to:
- Establish a correct implementation
- Create a solid foundation
- Enable future extensions
- Provide a fair comparison point

**Next:** Add news data and see how much it helps! 📰

---

## 📚 References

- Yahoo Finance: https://finance.yahoo.com/
- yfinance: https://github.com/ranaroussi/yfinance
- PyTorch: https://pytorch.org/

---

**Author:** Capstone Team  
**Contact:** [Your email]  
**Repository:** [Your repo]

---

_"Simple is better than complex. Complex is better than complicated."_ — The Zen of Python
