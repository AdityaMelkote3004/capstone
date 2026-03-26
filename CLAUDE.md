# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMGTFFF (Multi-Modal Graph Transformer for Federated Financial Forecasting) — a capstone project predicting next-day stock return direction (up/down) across 87 tickers in 9 sectors. Phases 0 (data audit) and 1 (baselines) are complete on `main` and `baseline` branches. Phase 2 (encoders), Phase 4 (graph + GAT), and Phase 5 (federated learning) are the next steps.

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

## Current State (as of 2026-03-26)

### Completed (Phase 0 + Phase 1)
- **Data audit**: `results/phase0_data_audit/data_statistics.json` — 26,603 rows, 87 tickers, 9 sectors, 50.2% target balance
- **Global date split** (literature-standard, in `src/data/stocknet_dataset.py`):
  - Train: 2014-01-02 to 2015-03-31 (15,969 rows, 15,534 sliding windows)
  - Val: 2015-04-01 to 2015-07-31 (4,359 rows, 3,924 windows)
  - Test: 2015-08-01 to 2015-12-31 (6,275 rows, 5,840 windows)
- **12 baseline experiments** (3 models x 4 feature sets) all near-random as expected:
  - Best: LSTM on FS2 (Price+Fundamentals), MCC=+0.042, Accuracy=52.1%
  - Results in `results/phase1_baselines/` (metrics.json per experiment + summary.json + breakdown.json)
- **Paper documentation**: `results/paper_results.md` — complete Phase 0+1 writeup ready for the research paper

### Key Data Pipeline Details
- `build_datasets(parquet_path, feature_set, window_size=5)` returns `(train_ds, val_ds, test_ds, info)`
- Split constants: `TRAIN_END`, `VAL_START`, `VAL_END`, `TEST_START` defined in `stocknet_dataset.py`
- `Trainer(model, train_dataset, val_dataset, test_dataset, ...)` — accepts explicit val set (no random carving)
- Dataset columns include: `Company_Texts`, `Event_Texts` (raw tweet strings), all features in `FEATURE_SETS`
- AGFS ticker has only 8 training rows (listed 2015-01-07) — exclude from per-ticker analysis if <50 rows
- `SECTOR_MAP` in stocknet_dataset.py maps 9 sector names to integer IDs 0-8

### Environment
- Python with PyTorch 2.8.0+cpu (NO CUDA/GPU)
- `transformers` NOT installed — must `pip install transformers` for FinBERT
- `torch_geometric` NOT installed — must `pip install torch_geometric` for GAT
- `networkx` IS installed (3.4.2)
- Platform: Windows 10, shell is bash

## Phase 2-5 Subagent Implementation Plan

The next step is to launch 3 parallel subagents, each simulating a teammate, each working in an isolated git worktree on its own branch. All 3 can run in parallel since they are independent until the fusion sync point.

### Subagent 1: Member 2 — FinBERT Text Encoder
**Branch:** `feat/text-encoder`
**Role:** NLP pipeline — tweet encoding + text-only baseline

**Tasks:**
1. `pip install transformers` (needed for ProsusAI/finbert)
2. Write `src/models/text_encoder.py`:
   - Load `ProsusAI/finbert` (AutoModel + AutoTokenizer from HuggingFace)
   - Takes raw tweet text (`Company_Texts`, `Event_Texts` columns from parquet)
   - Tokenize with FinBERT tokenizer (max 512 tokens)
   - Extract [CLS] embedding (768-d)
   - Project 768-d to 64-d via a learned linear layer
   - Handle empty/missing texts: use a learned default embedding or zero vector
3. Cache all embeddings to disk:
   - Process all 26,603 rows through FinBERT (slow on CPU — will take time)
   - Save as `data/processed/finbert_embeddings.pt` — tensor shape `[26603, 2, 64]` (company + event)
   - This cached file is loaded by the main pipeline later — FinBERT inference should only run once
4. Train a text-only classifier: FinBERT embeddings -> linear -> binary prediction
   - Use the same global date split (import `split_by_date` from stocknet_dataset.py)
   - Record standalone text Accuracy/F1/MCC to `results/phase2_encoders/text_encoder/metrics.json`
5. Generate t-SNE visualization of embeddings colored by Target (up/down)
   - Save to `results/phase2_encoders/text_encoder/tsne_embeddings.png`
6. Write a brief "text signal strength" paragraph for the paper

**Deliverables:**
- `src/models/text_encoder.py` — FinBERT encoder module
- `data/processed/finbert_embeddings.pt` — cached embeddings (gitignored, but script to regenerate)
- `scripts/cache_finbert_embeddings.py` — standalone script to regenerate the cache
- `results/phase2_encoders/text_encoder/` — metrics.json, tsne_embeddings.png

**Key integration details:**
- The text encoder's forward() should accept raw text strings and return `(batch, 64)` tensor
- OR accept pre-computed embeddings tensor and return projected `(batch, 64)` tensor
- Must handle the 54.9% of rows with zero tweets (empty strings) gracefully
- Output 64-d embedding will later be concatenated with temporal (64-d) and fundamental (32-d) embeddings for fusion

### Subagent 2: Member 3 — Temporal Encoder + Fundamental Encoder
**Branch:** `feat/temporal-fundamental`
**Role:** Numerical feature encoders — time-series transformer + fundamentals MLP

**Tasks (Phase 2B — Temporal Encoder):**
1. Write `src/models/temporal_encoder.py`:
   - Input: sliding window of W=5 trading days x K technical features per day
   - Use these features per timestep: Return, RSI_14, MACD, MACD_Hist, Volatility_5, Price_MA5_Ratio, Price_MA10_Ratio, Volume_Change, HL_Spread (pick ~8-10 most informative from PRICE_FEATURES)
   - Architecture: 2-layer Transformer encoder with positional encoding
     - d_model=64, nhead=4, dim_feedforward=128, dropout=0.2
   - Output: 64-d embedding (mean-pool over time steps, or use [CLS]-style learnable token)
2. Write windowing logic: for each (ticker, date), gather previous W days of features
   - NOTE: windowing is already handled by `StockNetDataset._build()` which creates `(W, F)` windows — reuse it
3. Train temporal-only classifier -> record Accuracy/F1/MCC
   - Use the global date split via `build_datasets()`
   - Save to `results/phase2_encoders/temporal_encoder/metrics.json`
4. Visualize attention weights: which days in the 5-day window get highest attention?
   - Save to `results/phase2_encoders/temporal_encoder/attention_heatmap.png`

**Tasks (Phase 2C — Fundamental Encoder):**
1. Write `src/models/fundamental_encoder.py`:
   - Input: 8 SEC EDGAR features from `FUNDAMENTAL_FEATURES` list + 5 derived ratios from `ALL_FUNDAMENTAL_COLS` = up to 13 features
   - Handle missing values: add a binary mask (1=present, 0=missing) as extra input features -> doubles input dims
   - Architecture: MLP — (input_dim*2) -> 64 -> ReLU -> Dropout -> 32
   - Output: 32-d embedding
2. Train fundamental-only classifier -> record metrics
   - Save to `results/phase2_encoders/fundamental_encoder/metrics.json`
3. Analyze feature importance (gradient-based)
   - Save to `results/phase2_encoders/fundamental_encoder/feature_importance.png`

**Deliverables:**
- `src/models/temporal_encoder.py` — Transformer encoder
- `src/models/fundamental_encoder.py` — MLP encoder
- `results/phase2_encoders/temporal_encoder/` — metrics, attention heatmap
- `results/phase2_encoders/fundamental_encoder/` — metrics, feature importance plot

**Key integration details:**
- Temporal encoder: input `(batch, 5, K)` -> output `(batch, 64)`
- Fundamental encoder: input `(batch, F)` with mask -> output `(batch, 32)`
- Both use the existing `Trainer` class (import from `src/training/trainer.py`) or a lightweight training loop
- Both use `build_datasets()` for the global date split — DO NOT create a new split

### Subagent 3: Member 4 — Graph Construction + GAT + Federated Learning
**Branch:** `feat/graph-federated`
**Role:** Graph structure + GAT + federated training framework

**Tasks (Phase 4 Prep — Graph Construction):**
1. `pip install torch_geometric` (or `pip install torch-scatter torch-sparse torch-geometric` if needed)
2. Write `src/models/graph_model.py`:
   - Build stock graph: 87 nodes (one per ticker), using `SECTOR_MAP` for sector IDs
   - **Static edges**: connect stocks in the same sector (from `Sector` column)
   - **Dynamic edges**: if two tickers co-occur in `Event_Texts` on the same day, add a temporary edge
   - **Correlation edges**: compute 20-day rolling return correlation between all pairs; add edge if |corr| > 0.5
   - Store as PyTorch Geometric `Data` object (edge_index + edge_attr)
3. Write a 2-layer GAT in the same file:
   - Input: node features (160-d from future fusion: 64 text + 64 temporal + 32 fundamental) + edge_index + edge_attr
   - 2-layer GAT, 4 attention heads
   - Output: graph-enriched 160-d per node
   - For now, test with dummy 160-d input to verify the GAT runs on the constructed graph without errors
4. Visualize the sector graph (networkx plot, colored by sector)
   - Save to `results/phase4_graph/graph_statistics/sector_graph.png`
5. Compute graph statistics: avg degree, clustering coefficient, diameter
   - Save to `results/phase4_graph/graph_statistics/graph_stats.json`

**Tasks (Phase 5 Prep — Federated Framework):**
1. Write `src/training/federated.py`:
   - `FederatedTrainer` class
   - Split data by `Sector` column -> 9 clients (one per sector)
   - Implement **FedAvg**: for each round, each client trains E=3 local epochs, then average model weights
   - Implement **FedProx**: add proximal term (mu/2)||w - w_global||^2 to local loss
   - Track per-round global accuracy on val set
2. Test with a simple MLP model to verify federation logic works before plugging in the full model
   - Use the existing `MLPBaseline` from `src/models/baselines.py` as the test model
   - Record results to `results/phase5_federated/federation_test/metrics.json`

**Deliverables:**
- `src/models/graph_model.py` — graph construction + GAT module
- `src/training/federated.py` — FedAvg + FedProx trainers
- `results/phase4_graph/graph_statistics/` — graph visualization, stats JSON
- `results/phase5_federated/federation_test/` — federation test metrics
- Working federation loop verified on dummy/simple model

**Key integration details:**
- Graph construction needs the parquet file for sector info and Event_Texts for dynamic edges
- GAT input will be 160-d (64+64+32 from the three encoders) — use random tensors for testing now
- Federated split is by SECTOR (9 clients), NOT by ticker
- The `Trainer` class can be adapted, but FederatedTrainer needs its own training loop
- Sector sizes are unequal (5-13 tickers per sector) — FedAvg should weight by client dataset size

### Launching the Subagents

To launch all 3 in a new session, use the Agent tool with `isolation: "worktree"` for each, running them in parallel in a single message. Each agent should:
1. Start from `main` branch (which equals `baseline`)
2. Create its own feature branch
3. Implement all tasks listed above
4. Commit and push to its branch
5. Report deliverables and metrics when done

After all 3 complete, merge each feature branch into main at the sync point.

### Fusion (Phase 3) — After All Encoders Are Done
After Members 2, 3, 4 finish, the next step is `src/models/mmgtfff.py` — the full MMGTFFF model:
- Concatenate: text_embedding (64-d) + temporal_embedding (64-d) + fundamental_embedding (32-d) = 160-d
- Pass through GAT for graph-enriched representation
- Final classifier head: 160-d -> 64 -> 2 (binary prediction)
- Train with FedAvg/FedProx using sector-based federation
- This is a sync point — all team members coordinate
