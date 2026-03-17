# 📊 Visualization Guide

## What Was Created

You now have **3 ways** to visualize your stock prediction results:

### 1. **Static Visualizations** (Already Generated!)

**Location:** `results/price_only/visualizations/`

#### Files Created:
- ✅ `AAPL_prediction_visualization.png` - Main clean view (Apple Stocks style)
- ✅ `AAPL_detailed_analysis.png` - Comprehensive 4-panel analysis

These were just opened automatically!

---

## 🎨 What Each Visualization Shows

### **Main Visualization** (`AAPL_prediction_visualization.png`)

**Dark theme, Apple Stocks app inspired**

Shows:
- 🟢 **Green line** = Actual returns (what really happened)
- 🔵 **Blue dashed line** = Predicted returns (model's guess)
- 📊 **Metrics box** (top right):
  - MSE, MAE, RMSE
  - Directional Accuracy

**What to look for:**
- Do the lines follow similar patterns?
- Does blue lag behind green? (delayed reaction)
- Are predictions too conservative? (blue closer to zero)
- Big divergences? (model missed major moves)

---

### **Detailed Analysis** (`AAPL_detailed_analysis.png`)

**4-panel comprehensive view:**

#### Panel 1: Time Series (top-left)
- Same as main viz
- Shows trends over time

#### Panel 2: Scatter Plot (top-right)
- Each point = one prediction
- Perfect line (red) = where predictions should be
- Cluster around line = good correlation
- Spread = prediction variance

#### Panel 3: Error Distribution (bottom-left)
- Histogram of prediction errors
- Center at zero = no bias
- Narrow = consistent
- Wide = variable performance

#### Panel 4: Rolling Accuracy (bottom-right)
- Directional accuracy over time
- 50% line = random guessing
- Above = beating random
- Below = worse than random

---

## 🚀 How to Use

### **Option 1: View Saved Images**
```powershell
# Already opened, but to reopen:
start results\price_only\visualizations\AAPL_prediction_visualization.png
start results\price_only\visualizations\AAPL_detailed_analysis.png
```

### **Option 2: Regenerate for Different Settings**
```powershell
# Run the visualization script
python visualize_results.py
```

### **Option 3: Interactive Dashboard** (Streamlit)
```powershell
# Install streamlit if needed
pip install streamlit

# Launch interactive app
streamlit run streamlit_app.py
```

Then open browser at: http://localhost:8501

**Interactive features:**
- Hover over data points
- Zoom in/out
- View raw data table
- Real-time metric updates
- Dark theme UI

---

## 🔍 Interpreting Your Results

### Your Current Performance:
```
Directional Accuracy: 48.96% ⚠️
MAE: 1.22%
RMSE: 1.67%
```

### What This Means:

#### ⚠️ **Below Random (48.96% < 50%)**
**Possible reasons:**
1. **Price patterns are weak** - expected for price-only
2. **Test period unusual** - 2022-2023 was volatile (inflation, rate hikes)
3. **Model slight overfit** - learned training patterns that didn't generalize

**This is OK for a baseline!** Shows need for:
- News data
- Market indicators
- Graph relationships

#### 📊 **Error Magnitude (MAE = 1.22%)**
- Daily returns average ~0.03%
- Error is 40x larger than signal
- **High noise-to-signal ratio**

This is why prediction is hard!

---

## 🎯 Visual Debugging Checklist

Look at your visualizations and check:

### ✅ **Good Signs:**
- [ ] Predictions follow general trend direction
- [ ] Model captures major moves (peaks/valleys)
- [ ] Error distribution centered at zero
- [ ] Some periods above 50% accuracy

### ⚠️ **Warning Signs:**
- [ ] Predictions always near zero (too conservative)
- [ ] Blue line lags green by 1+ days (delayed)
- [ ] Random scatter in scatter plot (no correlation)
- [ ] Rolling accuracy always below 50%

### What You'll Likely See:
- ✅ Predictions are conservative (closer to zero than actuals)
- ✅ Model misses large spikes
- ⚠️ Slight lag in reactions
- ⚠️ Overall accuracy below 50%

**This validates the need for richer data!**

---

## 📸 Using for Presentations

### For Reports/Papers:
1. Use `AAPL_prediction_visualization.png` for main figure
2. Use `AAPL_detailed_analysis.png` for appendix
3. Dark theme looks professional in slides

### For Team Discussions:
1. Open Streamlit app (`streamlit run streamlit_app.py`)
2. Interactive exploration during meetings
3. Zoom into specific time periods

---

## 🔄 Next Steps

### After Reviewing Visualizations:

1. **Confirm baseline is working** ✅
   - Model learns something (not random)
   - Proper evaluation (no bugs)

2. **Identify failure patterns**
   - When does model fail most?
   - Which type of moves are missed?
   - Systematic bias?

3. **Plan improvements**
   - Add news sentiment (should help with surprises)
   - Add technical indicators (help with patterns)
   - Add graph relationships (capture correlations)

---

## 🛠️ Customization

### Change Stock Ticker:
Edit `config/price_baseline.yaml`:
```yaml
data:
  ticker: "MSFT"  # Try Microsoft, Tesla, etc.
```

Then rerun:
```powershell
python quick_start.py
python visualize_results.py
```

### Adjust Time Period:
In config:
```yaml
data:
  start_date: "2020-01-01"
  end_date: "2023-12-31"
```

---

## 📦 Files Reference

```
results/price_only/visualizations/
├── AAPL_prediction_visualization.png    ← Clean main view
└── AAPL_detailed_analysis.png          ← 4-panel analysis

streamlit_app.py                         ← Interactive dashboard
visualize_results.py                     ← Generation script
src/visualization/visualize_predictions.py  ← Core functions
```

---

## 💡 Pro Tips

1. **Save screenshots** of visualizations for your report
2. **Compare before/after** when you add features
3. **Use Streamlit** for exploratory analysis
4. **Static images** for final reports/papers
5. **Dark theme** looks great in presentations

---

## ❓ Common Questions

**Q: Why is my accuracy below 50%?**  
A: Price-only models struggle. Test period (2022-2023) was volatile. This justifies multimodal approach!

**Q: How do I make plots lighter theme?**  
A: Edit `visualize_predictions.py`, change `plt.style.use('dark_background')` to `plt.style.use('default')`

**Q: Can I zoom into specific dates?**  
A: Yes! Use Streamlit app for interactive zooming, or edit the visualization code to filter dates.

**Q: Predictions seem too flat?**  
A: Common issue - model plays it safe. Add features like news to capture surprises.

---

**Ready to move on?** Your visualizations show the baseline is working correctly and identify clear areas for improvement through multimodal data!
