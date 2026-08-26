# Phase 2 — FinBERT Sentiment Ablation

## Objective

Test whether FinBERT's 3-class financial sentiment output is sufficient for stock prediction compared with the previously used 768-D FinBERT representation.

## Representation

Each text input is passed through ProsusAI/FinBERT's sequence-classification model. The classification logits are converted with softmax into three probabilities:

- Positive
- Neutral
- Negative

These probabilities form a 3-D sentiment representation.

## Experiments

| Experiment | Input |
|---|---|
| A | Price |
| B | Price + Fundamentals |
| C | Price + FinBERT Sentiment |
| D | Price + Fundamentals + FinBERT Sentiment |

## Validation Results

| Experiment | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| A_Price | 0.5454 | 0.5642 | 0.0952 | 0.5543 |
| D_Price_Fundamentals_FinBERT_Sentiment | 0.5364 | 0.5532 | 0.0767 | 0.5507 |
| B_Price_Fundamentals | 0.5354 | 0.5529 | 0.0748 | 0.5482 |
| C_Price_FinBERT_Sentiment | 0.5301 | 0.5796 | 0.0712 | 0.5500 |

## Selected Configuration

**A_Price** was selected using validation MCC, with AUC as the secondary criterion.

## Final Test Result

| Metric | Value |
|---|---:|
| ACCURACY | 0.5031 |
| F1 | 0.5233 |
| MCC | 0.0091 |
| AUC | 0.5186 |

## Previous 768-D FinBERT LSTM References

| Feature Set | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Price + Company FinBERT | 0.5248 | 0.4721 | 0.0459 | 0.5304 |
| Price + Fundamentals + Company FinBERT | 0.4870 | 0.6550 | 0.0000 | 0.5270 |

## Interpretation

If the 3-D sentiment representation performs similarly to the 768-D embedding, the simpler and more interpretable representation is preferable. If the 768-D embedding performs substantially better, the model is using information beyond sentiment.

## Next Step

Use this result to decide whether the centralized graph model should use simple FinBERT sentiment features or a richer text representation. Do not add graph structure or federated learning until the centralized baseline is established.
