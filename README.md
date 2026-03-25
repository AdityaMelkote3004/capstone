# Baseline Logistic Regression (LR Baseline)

A baseline implementation of Logistic Regression for stock price prediction using multi-modal data sources including price data, sentiment analysis, and fundamental indicators.

## Project Overview

This project builds and evaluates logistic regression models for binary stock price prediction (up/down classification) using a progressive feature engineering approach:
- **Step 1**: Price data only
- **Step 2**: Price + Sentiment features
- **Step 3**: Price + Fundamentals features
- **Step 4**: Full structured data (Price + Sentiment + Fundamentals)

## Project Structure

```
BaselineLR/
├── data_loader.py              # Data loading and preprocessing utilities
├── trainer.py                  # Model training and evaluation logic
├── run_baseline.py             # Main script to run the baseline model
├── results_saver.py            # Utilities for saving results and metrics
├── visualiser.py               # Visualization functions for results
├── streamlit_app.py            # Streamlit web interface (main)
├── streamlit_app1.py           # Alternative Streamlit app
├── stocknet_final_modeling_set.csv  # Input dataset
├── requirements.txt            # Python dependencies
├── results/                    # Output directory
│   └── phase1_baselines/
│       └── logistic_regression/
│           ├── config.yaml     # Model configuration
│           ├── metrics.json    # Performance metrics
│           ├── summary_table.csv
│           ├── confusion_matrices.png
│           ├── feature_importance.png
│           ├── metrics_comparison.png
│           ├── class_distribution.png
│           ├── per_sector_heatmap.png
│           └── per_sector/     # Per-sector results
│               ├── step1_price_only.csv
│               ├── step2_price_sentiment.csv
│               ├── step3_price_fundamentals.csv
│               └── step4_full_structured.csv
└── __pycache__/                # Python cache
```

## Features

- **Progressive Feature Engineering**: Compare model performance across different feature combinations
- **Sector-wise Analysis**: Evaluate model performance across different stock sectors
- **Multiple Metrics**: Accuracy, F1-score, precision, recall, and more
- **Visualizations**: Confusion matrices, feature importance, and performance comparisons
- **Web Interface**: Streamlit app for interactive exploration of results
- **Configuration Management**: YAML-based configuration for easy experimentation

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- streamlit
- pyyaml
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AdityaMelkote3004/capstone.git
cd capstone
git checkout LR-baseline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Prerequisites
- Python 3.8 or higher installed
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/AdityaMelkote3004/capstone.git
cd capstone
git checkout LR-baseline
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas
- scikit-learn
- numpy
- streamlit
- pyyaml
- matplotlib
- seaborn

## Usage

### Option 1: Train the Baseline Model

To train and evaluate the baseline logistic regression model from scratch:

```bash
python run_baseline.py
```

**What happens:**
1. Loads the dataset from `stocknet_final_modeling_set.csv`
2. Splits data chronologically:
   - **Train**: 2014-01-01 to 2015-03-31
   - **Val**: 2015-04-01 to 2015-07-31
   - **Test**: 2015-08-01 to 2015-12-31
3. Trains logistic regression models with 4 progressive feature sets:
   - Step 1: Price features only (15 features)
   - Step 2: Price + Sentiment (18 features)
   - Step 3: Price + Fundamentals (43 features)
   - Step 4: Full structured data (46 features)
4. Generates metrics (Accuracy, F1-Score, MCC, AUC-ROC)
5. Creates visualizations and saves results

**Estimated runtime:** 5-10 minutes

**Output location:** `results/phase1_baselines/logistic_regression/`

### Option 2: View Results with Streamlit (Interactive Dashboard)

If you've already trained the model or want to explore existing results:

```bash
streamlit run streamlit_app.py
```

**In your browser:**
- Opens automatically at `http://localhost:8501`
- If not, manually navigate to that URL

**Features available:**
- 📊 Visual exploration of all model metrics
- 🏭 Sector-wise performance analysis
- 🔲 Confusion matrices for each step
- 📈 Feature importance plots
- 📋 Comparison tables across feature engineering steps
- 💾 Export capabilities for results

**To stop the Streamlit server:**
Press `Ctrl+C` in the terminal

### Option 3: Alternative Streamlit App

An alternative interface is also available:

```bash
streamlit run streamlit_app1.py
```

## Viewing Results

If you only want to view pre-computed results **without retraining**:

Results are already saved in: `results/phase1_baselines/logistic_regression/`

**Key files:**
- `metrics.json` — All model metrics in JSON format
- `summary_table.csv` — Quick comparison table
- `config.yaml` — Model configuration and parameters
- `confusion_matrices.png` — Visual confusion matrices
- `feature_importance.png` — Feature importance ranking
- `metrics_comparison.png` — Performance metric comparisons
- `per_sector/` — Detailed results per stock sector

## Data Input

The dataset should be a CSV file (`stocknet_final_modeling_set.csv`) with the following structure:
- Stock price data (open, high, low, close)
- Sentiment features (from news/social media)
- Fundamental indicators (financial ratios, earnings, etc.)
- Sector information
- Binary target variable (1=price up, 0=price down)

## Output Format

Results are saved in `results/phase1_baselines/logistic_regression/`:

- **metrics.json**: Dictionary of all performance metrics
- **summary_table.csv**: Overview of all results
- **config.yaml**: Model configuration and hyperparameters
- **Visualizations**: PNG files for confusion matrices, feature importance, and metrics
- **per_sector/**: CSV files with per-sector performance metrics for each feature engineering step

## Key Results

The model provides performance metrics across:
1. Different feature engineering steps (4 stages)
2. Multiple stock sectors
3. Standard classification metrics (accuracy, precision, recall, F1-score)
4. ROC-AUC and other advanced metrics

## Configuration

Edit `results/phase1_baselines/logistic_regression/config.yaml` to adjust:
- Model hyperparameters
- Feature sets
- Train/test split ratios
- Random seed for reproducibility

## License

This project is part of the capstone project.

## Contact

For questions or issues, contact the project maintainers.
