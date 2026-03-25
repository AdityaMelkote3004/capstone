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

## Usage

### Run Baseline Model

To train and evaluate the baseline logistic regression model:

```bash
python run_baseline.py
```

This will:
1. Load the dataset from `stocknet_final_modeling_set.csv`
2. Train logistic regression models with progressive feature sets
3. Generate metrics and visualizations
4. Save results to `results/phase1_baselines/logistic_regression/`

### View Results with Streamlit

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

This provides:
- Visual exploration of model metrics
- Sector-wise performance analysis
- Confusion matrices and feature importance plots
- Comparison of different feature engineering steps

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
