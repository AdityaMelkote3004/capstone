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

Company text is available for approximately 43.4% of stock-days.

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
```

For stock-days containing multiple texts, the embeddings are mean-pooled:

```text
Text 1 ─┐
Text 2 ─┤
Text 3 ─┤ → FinBERT → 768-d embeddings → Mean Pooling
  ... ──┘
```

This produces one 768-dimensional representation per stock-day.

For stock-days without text, a zero vector is used.

A `Company_Text_Present` indicator is also included.

Generated embedding artifacts:

```text
company_embeddings.npy
event_embeddings.npy
```

These are intentionally excluded from Git because they are generated artifacts.

---

# 3. Phase 2 Feature Sets

The first experiments focus on Company Text.

## FS5 — Price + Company FinBERT

```text
14 price / technical features
+ 768 company FinBERT dimensions
+ 1 text-presence indicator

= 783 features
```

## FS6 — Price + Fundamentals + Company FinBERT

```text
14 price / technical features
+ 8 fundamental features
+ 768 company FinBERT dimensions
+ 1 text-presence indicator

= 791 features
```

The original Phase 1 feature sets were not modified.

---

# 4. Experimental Setup

Three models were evaluated for each feature set.

### Logistic Regression

- `C = 1.0`
- `max_iter = 1000`

### LSTM

- 2 layers
- Hidden dimension = 64
- Dropout = 0.2
- Window size = 5
- Learning rate = 0.001

### MLP

- Hidden dimension = 128
- Dropout = 0.2
- Learning rate = 0.001

### Training

- Seed = 42
- Maximum epochs = 50
- Early stopping patience = 10

### Metrics

- Accuracy
- F1
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

---

# 5. Dataset Sizes

Both Phase 2 feature sets use the same split:

| Split | Rows | Model samples |
|---|---:|---:|
| Train | 15,969 | 15,534 |
| Validation | 4,359 | 3,924 |
| Test | 6,275 | 5,840 |

The reduction from rows to model samples comes from constructing 5-day windows independently for each ticker.

---

# 6. Phase 2 Results

## FS5 — Price + Company FinBERT

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.4920 | 0.5589 | -0.0080 | 0.4953 |
| **LSTM** | **0.5248** | 0.4721 | **0.0459** | **0.5304** |
| MLP | 0.4830 | **0.6109** | -0.0221 | 0.5053 |

### Logistic Regression

```text
Accuracy = 0.4920
F1       = 0.5589
MCC      = -0.0080
AUC      = 0.4953
```

### LSTM

```text
Accuracy = 0.5248
F1       = 0.4721
MCC      = 0.0459
AUC      = 0.5304
```

Training:

```text
Epoch 1:
Train Acc = 0.510
Val Acc   = 0.484
Val MCC   = 0.0000

Epoch 5:
Train Acc = 0.516
Val Acc   = 0.479
Val MCC   = -0.0274

Epoch 10:
Train Acc = 0.521
Val Acc   = 0.484
Val MCC   = -0.0075

Early stopping at epoch 14
```

### MLP

```text
Accuracy = 0.4830
F1       = 0.6109
MCC      = -0.0221
AUC      = 0.5053
```

Training:

```text
Epoch 1:
Train Acc = 0.501
Val Acc   = 0.481
Val MCC   = -0.0215

Epoch 5:
Train Acc = 0.525
Val Acc   = 0.489
Val MCC   = -0.0001

Epoch 10:
Train Acc = 0.541
Val Acc   = 0.490
Val MCC   = -0.0009

Epoch 15:
Train Acc = 0.561
Val Acc   = 0.491
Val MCC   = 0.0008

Epoch 20:
Train Acc = 0.583
Val Acc   = 0.492
Val MCC   = 0.0013

Early stopping at epoch 24
```

---

## FS6 — Price + Fundamentals + Company FinBERT

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.4932 | 0.5606 | -0.0053 | 0.4961 |
| LSTM | 0.4870 | **0.6550** | 0.0000 | 0.5270 |
| MLP | **0.5038** | 0.5506 | **0.0140** | 0.5124 |

### Logistic Regression

```text
Accuracy = 0.4932
F1       = 0.5606
MCC      = -0.0053
AUC      = 0.4961
```

### LSTM

```text
Accuracy = 0.4870
F1       = 0.6550
MCC      = 0.0000
AUC      = 0.5270
```

Training:

```text
Epoch 1:
Train Acc = 0.512
Val Acc   = 0.484
Val MCC   = -0.0101

Epoch 5:
Train Acc = 0.514
Val Acc   = 0.484
Val MCC   = 0.0000

Epoch 10:
Train Acc = 0.514
Val Acc   = 0.483
Val MCC   = -0.0130

Early stopping at epoch 13
```

### MLP

```text
Accuracy = 0.5038
F1       = 0.5506
MCC      = 0.0140
AUC      = 0.5124
```

Training:

```text
Epoch 1:
Train Acc = 0.508
Val Acc   = 0.480
Val MCC   = -0.0241

Epoch 5:
Train Acc = 0.519
Val Acc   = 0.482
Val MCC   = -0.0247

Epoch 10:
Train Acc = 0.539
Val Acc   = 0.524
Val MCC   = 0.0531

Epoch 15:
Train Acc = 0.560
Val Acc   = 0.488
Val MCC   = -0.0114

Epoch 20:
Train Acc = 0.583
Val Acc   = 0.485
Val MCC   = -0.0163

Early stopping at epoch 20
```

---

# 7. Complete Phase 2 Results

| Feature Set | Model | Accuracy | F1 | MCC | AUC |
|---|---|---:|---:|---:|---:|
| Price + Company FinBERT | Logistic Regression | 0.4920 | 0.5589 | -0.0080 | 0.4953 |
| Price + Company FinBERT | **LSTM** | **0.5248** | 0.4721 | **0.0459** | **0.5304** |
| Price + Company FinBERT | MLP | 0.4830 | **0.6109** | -0.0221 | 0.5053 |
| Price + Fundamentals + Company FinBERT | Logistic Regression | 0.4932 | 0.5606 | -0.0053 | 0.4961 |
| Price + Fundamentals + Company FinBERT | LSTM | 0.4870 | **0.6550** | 0.0000 | 0.5270 |
| Price + Fundamentals + Company FinBERT | MLP | **0.5038** | 0.5506 | **0.0140** | 0.5124 |

---

# 8. Phase 1 vs Phase 2

The strongest Phase 1 result was:

```text
Price + Fundamentals + LSTM

MCC = 0.042
AUC = 0.532
```

The strongest Phase 2 result by MCC was:

```text
Price + Company FinBERT + LSTM

MCC = 0.0459
AUC = 0.5304
```

### Direct comparison

| Experiment | Model | Accuracy | F1 | MCC | AUC |
|---|---|---:|---:|---:|---:|
| Phase 1 — Price + Fundamentals | LSTM | — | — | **0.042** | **0.532** |
| Phase 2 — Price + Company FinBERT | LSTM | **0.5248** | 0.4721 | **0.0459** | 0.5304 |
| Phase 2 — Price + Fundamentals + Company FinBERT | LSTM | 0.4870 | **0.6550** | 0.0000 | 0.5270 |

The Phase 2 Price + Company FinBERT LSTM has a slightly higher MCC than the Phase 1 best model, but slightly lower AUC.

This difference is small and should not be interpreted as a demonstrated improvement without further statistical testing.

---

# 9. Key Findings

### Finding 1 — FinBERT does not produce a large improvement by simple concatenation

Adding the 768-dimensional company embedding does not substantially outperform the structured baseline.

The best Phase 2 result is roughly comparable to the best Phase 1 result.

### Finding 2 — Company FinBERT + LSTM is the strongest Phase 2 configuration

Among the six Phase 2 experiments:

```text
Price + Company FinBERT + LSTM

MCC = 0.0459
AUC = 0.5304
```

is the strongest combination according to MCC and AUC.

### Finding 3 — Adding fundamentals to the FinBERT representation did not help the LSTM

The LSTM changed from:

```text
Price + Company FinBERT
MCC = 0.0459
```

to:

```text
Price + Fundamentals + Company FinBERT
MCC = 0.0000
```

So adding the eight fundamental features did not improve this particular configuration.

### Finding 4 — F1 alone can be misleading

The FS6 LSTM has:

```text
F1 = 0.6550
```

but:

```text
MCC = 0.0000
AUC = 0.5270
```

Therefore, the high F1 should not be interpreted as strong predictive ability.

MCC and AUC provide a more useful picture of the model's discriminative performance here.

### Finding 5 — The text representation remains very sparse

Company text exists for only:

```text
11,550 / 26,603 stock-days
```

while:

```text
15,053 / 26,603 stock-days
```

contain no company text.

This limits how frequently the model can exploit semantic information.

---

# 10. Limitations

The current experiment uses:

```text
Individual text
      ↓
FinBERT [CLS]
      ↓
Mean pooling
      ↓
768-dimensional daily representation
```

This is intentionally a simple baseline.

Potential limitations include:

1. Mean pooling can remove important differences between texts.
2. Conflicting positive and negative information can be averaged together.
3. More than half of company stock-days contain no text.
4. The 768-dimensional representation substantially increases model input dimensionality.
5. FinBERT is frozen and is not fine-tuned for the stock-direction task.
6. Temporal alignment of individual text with the stock-day still needs to be explicitly validated.

---

# 11. Completed

- [x] Audited text availability
- [x] Loaded FinBERT
- [x] Generated company embeddings
- [x] Generated event embeddings
- [x] Verified embedding/dataframe alignment
- [x] Created Phase 2 modeling dataset
- [x] Added FS5 and FS6
- [x] Preserved Phase 1 feature sets
- [x] Ran Logistic Regression
- [x] Ran LSTM
- [x] Ran MLP
- [x] Saved all six experiment results
- [x] Compared Phase 2 against Phase 1

---

# 12. Next Steps

Before increasing model complexity, the next steps are:

1. Validate the temporal semantics of `Company_Texts` and `Event_Texts`.
2. Run the equivalent Event FinBERT experiments.
3. Investigate alternative text aggregation methods.
4. Test dimensionality reduction / learned projection of the 768-dimensional embeddings.
5. Investigate richer multimodal fusion.
6. Move toward graph-based modeling.
7. Eventually investigate federated learning.

---

# 13. Repository Artifacts

Results are stored under:

```text
results/phase2_embeddings/
```

Structure:

```text
results/phase2_embeddings/
├── summary.json
├── full_results.json
├── FS5_Price_CompanyEmbedding/
│   ├── logistic_regression/
│   ├── lstm/
│   └── mlp/
└── FS6_Price_Fundamentals_CompanyEmbedding/
    ├── logistic_regression/
    ├── lstm/
    └── mlp/
```

Large generated artifacts are intentionally excluded from Git:

```text
company_embeddings.npy
event_embeddings.npy
dataset/stocknet_final_modeling_set_phase2.parquet
```

---

# Conclusion

Phase 2 successfully establishes a **FinBERT-based company-text baseline** on top of the existing StockNet price/technical features.

The current results show that simple frozen FinBERT embeddings with mean pooling provide **roughly comparable but not substantially better performance** than the existing structured baseline.

The current Phase 2 benchmark to beat is:

```text
Price + Company FinBERT + LSTM

Accuracy = 0.5248
F1       = 0.4721
MCC      = 0.0459
AUC      = 0.5304
```

The next phase should focus on validating temporal alignment and improving the text representation and fusion strategy rather than immediately increasing model complexity.
