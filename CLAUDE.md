# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMGTFFF (Multi-Modal Graph Transformer for Federated Financial Forecasting) — a capstone project predicting next-day stock return direction (up/down) across 87 tickers in 9 sectors. Phases 0 (data audit) and 1 (baselines) are complete; phases 2-6 will add tweet sentiment, SEC filings, graph transformers, and federated learning.

## Common Commands

```bash
# Run Streamlit dashboard (main entry point for viewing results)
streamlit run streamlit_app.py

# Quick start: download data, train, evaluate in one step
python quick_start.py

# Train Phase 1 baselines (Logistic Regression, LSTM, MLP × 4 feature sets)
python scripts/train_baselines.py

# Evaluate baselines and generate summary reports
python scripts/evaluate_baselines.py

# Per-ticker breakdown analysis
python scripts/evaluate_per_ticker.py

# Generate static prediction visualizations
python visualize_results.py

# Train price-only LSTM baseline from config
python -m src.training.train_price_model --config config/price_baseline.yaml
```

No test framework is configured. No linter or formatter is enforced.

## Architecture

### Two Model Families

1. **Regression models** (`src/models/price_model.py`): LSTMPriceModel and MLPPriceModel predict scalar returns. Trained via `src/training/train_price_model.py` with MSE loss.
2. **Classification models** (`src/models/baselines.py`): LSTMBaseline and MLPBaseline output 2-class logits (up/down). Trained via `src/training/trainer.py` with CrossEntropyLoss. Logistic Regression runs in `scripts/train_baselines.py` via sklearn.

### Data Pipeline

`src/data/price_loader.py` downloads Yahoo Finance data → computes next-day returns → creates sliding windows. `src/data/splits.py` performs time-based chronological splitting (never random). `src/data/stocknet_dataset.py` wraps the 87-ticker parquet dataset (`dataset/stocknet_final_modeling_set.parquet`) as a PyTorch Dataset with four feature set configurations.

### Feature Set System (defined in `src/data/stocknet_dataset.py`)

- **FS1_Price**: 14 technical indicators (Return, RSI, MACD, Volatility, etc.)
- **FS2_Price_Fundamentals**: FS1 + 8 SEC EDGAR fundamentals
- **FS3_Price_Tweets**: FS1 + 3 tweet count columns
- **FS4_Full_Structured**: All features combined

### Results Flow

Training scripts write JSON results to `results/phase1_baselines/`. The Streamlit app (`streamlit_app.py`) reads these JSON files to render metrics, confusion matrices, and per-ticker/per-sector breakdowns.

## Critical Conventions

- **No future information leakage**: All splits are chronological (time-based), never random. Normalization statistics come from training data only.
- **Seed management**: Use `src/utils/seed.py:set_seed()` which covers Python random, NumPy, PyTorch, and CUDA determinism.
- **Config-driven training**: Model hyperparameters and data params live in `config/*.yaml`. Training scripts load these configs.
- **Sliding window format**: Models expect input shape `(batch, window_size, features)` for LSTM or `(batch, features)` for MLP.
- **Primary metric for classification**: MCC (Matthews Correlation Coefficient) — used for early stopping in `src/training/trainer.py` with patience=10.
- **Data files are gitignored**: Raw/processed data in `data/`, checkpoints (*.pt, *.pth), and prediction outputs are not committed.
