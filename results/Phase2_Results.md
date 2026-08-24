# Phase 2 — FinBERT Embedding Baseline Results

## Objective

Phase 2 extends the existing StockNet baseline by replacing the crude tweet-count representation with semantic representations generated using FinBERT.

The objective is to determine whether pretrained financial-language embeddings provide additional predictive signal for next-day stock direction.

---

# 1. Dataset

- 26,603 stock-day observations
- 87 tickers
- 5-day sliding window
- Chronological train / validation / test split
- Train-only normalization
- Binary next-day direction target

The original Phase 1 dataset was kept unchanged.

### Text availability

| Modality | Rows with text | Rows without text | Mean texts/row | Max texts/row |
|---|---:|---:|---:|---:|
| Company Text | 11,550 | 15,053 | 1.96 | 553 |
| Event Text | 2,970 | 23,633 | 0.19 | 36 |

Company text is therefore available for approximately **43.4%** of stock-days.

---

# 2. FinBERT Embedding Generation

Model:

**ProsusAI/finbert**

For each individual text:

```text
Text
 ↓
FinBERT tokenizer
 ↓
FinBERT encoder
 ↓
[CLS] representation
 ↓
768-dimensional embedding
