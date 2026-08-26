# Stock Price Prediction — Research Roadmap

## 1. Goal

Build a leakage-free next-day stock movement prediction system using:

- Stock prices / technical indicators
- Company fundamentals
- Social/news text
- Events
- Relationships between companies

Primary prediction:

> Will the stock go **UP or DOWN tomorrow**?

Secondary prediction later:

> What is the expected next-day return?

The main metrics are:

- Accuracy
- Precision
- Recall
- Macro F1
- MCC
- ROC-AUC
- PR-AUC

For the trading side:

- Annualized return
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Turnover
- Transaction costs

---

# 2. What We Have Learned So Far

Our current best result is approximately:

- MCC: **0.0459**
- AUC: **0.5304**

The FinBERT experiments with:

- 768 dimensions
- 128 dimensions
- 64 dimensions
- VADER
- 3-dimensional sentiment output

did **not** produce a drastic improvement.

### Conclusion

The main problem is probably **not the dimensionality or choice of sentiment model**.

The bigger issues are:

1. Tweets are being mean-pooled too early.
2. Temporal windows need to be guaranteed consecutive trading days.
3. We need an exact StockNet-compatible benchmark.
4. Fundamentals need better point-in-time handling.
5. Missing fundamental values should not simply become zero.
6. We currently do not exploit relationships between stocks.
7. Fusion between modalities needs to be more intelligent.

Therefore:

> **Stop experimenting randomly with sentiment encoders. Fix the architecture and methodology first.**

---

# 3. Primary Literature Benchmark

## MAN-SF

The most relevant benchmark for our dataset is the MAN-SF paper because it evaluates on StockNet.

Reported results are approximately:

- Accuracy: **60.8%**
- F1: **60.5%**
- MCC: **0.195**
- Sharpe: **~1.05**

The architecture combines:

- Price sequence
- Social-media text
- Hierarchical attention
- Bilinear fusion
- Graph Attention Network (GAT)

This should be our first serious benchmark to reproduce.

## Why MAN-SF matters

Its ablation results show that architecture matters much more than simply changing the text embedding:

| Model/component | MCC |
|---|---:|
| LSTM + price | ~0.002 |
| GRU + social text | ~0.077 |
| GCN + price | ~0.093 |
| GRU + USE text | ~0.101 |
| GAT + price | ~0.117 |
| Concatenation | ~0.156 |
| Attention fusion | ~0.173 |
| **Bilinear fusion** | **~0.195** |

This strongly motivates:

> hierarchical text encoding + temporal attention + bilinear fusion + GAT.

---

# 4. Exact Development Plan

## Phase 0 — Freeze the evaluation methodology

Before building anything else, establish two evaluation protocols.

### Protocol A — Exact StockNet reproduction

Use:

- StockNet labels
- StockNet temporal split
- StockNet trading-day alignment
- StockNet-style lag window

This is the benchmark we can legitimately compare against published StockNet results.

### Protocol B — Our extended dataset

Keep our current dataset/split as a separate experiment.

Do **not** call the current split an exact StockNet reproduction unless it matches the original protocol.

---

# 5. Fix the Data Pipeline First

## 5.1 Trading-day alignment

Do NOT create sequences by simply taking five CSV rows.

We need:

```text
Trading Day 1
Trading Day 2
Trading Day 3
Trading Day 4
Trading Day 5
```

For each stock.

The window should represent actual consecutive trading days.

If there are no tweets on a trading day:

```text
tweet representation = zero
text_present = 0
```

The day itself must still exist in the sequence.

---

# 6. Prevent Look-Ahead Leakage

Every feature must be based only on information available before the prediction cutoff.

For every piece of information track:

```text
event/publication timestamp
availability timestamp
trading date
```

Example:

If news appears after market close:

```text
Monday 4:30 PM
```

it should not be used to predict Monday's close.

It becomes available for the next appropriate prediction period.

The same applies to fundamentals.

---

# 7. Fundamentals

Current fundamentals include things such as:

- Revenue
- Net income
- Assets
- Liabilities
- Equity
- EPS
- Cash
- ROA
- Debt/Equity
- Margins

The current preprocessing has substantial missingness and currently uses forward-fill/zero-fill.

We should change this.

## Do NOT simply do:

```text
missing → 0
```

Instead use:

```text
value
+
missing flag
+
days since last filing
```

Example:

```text
EPS = 0
EPS_missing = 0
```

means actual EPS is zero.

Whereas:

```text
EPS = 0
EPS_missing = 1
```

means the value was unavailable.

---

# 8. Use Fundamental Changes, Not Just Raw Values

Raw values such as:

```text
Revenue = $10B
```

are often not very useful for next-day movement.

Add features such as:

```text
Revenue growth
EPS growth
ROA change
Margin change
Debt/equity change
Cash change
```

Also include:

```text
days_since_filing
new_filing
```

Eventually include:

```text
earnings surprise
revenue surprise
guidance change
```

if reliable point-in-time data is available.

---

# 9. Model Architecture — Version 1

Do NOT immediately build the huge final model.

First reproduce a proven architecture.

## Overall flow

```text
                    STOCKNET DATA
                         |
                         v
              POINT-IN-TIME CLEANING
                         |
                         v
              TRADING-DAY ALIGNMENT
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
      PRICE         FUNDAMENTALS        TWEETS
        |                |                |
        v                v                v
      Bi-GRU            MLP          Tweet Encoder
        |                |                |
 Temporal Attention      |          Tweet Attention
        |                |                |
        |                |          Daily Tweet Vector
        |                |                |
        |                |            Daily GRU
        |                |                |
        |                |       Temporal Attention
        |                |                |
        +----------------+----------------+
                         |
                         v
                  BILINEAR FUSION
                         |
                         v
                STOCK REPRESENTATION
                         |
                         v
                INTER-STOCK GRAPH
                         |
                         v
                    MULTI-HEAD GAT
                         |
                         v
                     UP / DOWN
```

---

# 10. Price Encoder

## First choice

Use:

**GRU + temporal attention**

This follows the proven MAN-SF architecture.

Input:

- OHLCV
- technical indicators
- historical returns

Example:

```text
5–20 trading days
       |
       v
      GRU
       |
       v
Temporal attention
       |
       v
Price embedding
```

The attention mechanism lets the model learn which previous days matter most.

Do NOT start with CNN + Transformer.

We can test those later.

---

# 11. Text Encoder

## Major change from our current approach

Current approach:

```text
Tweet 1
Tweet 2
Tweet 3
...
    |
 FinBERT
    |
 mean pooling
    |
 one vector
```

This destroys too much information.

## New approach

```text
                 TODAY'S TWEETS
                       |
          +------------+------------+
          |            |            |
       Tweet 1      Tweet 2      Tweet N
          |            |            |
       Encoder      Encoder      Encoder
          |            |            |
          +------------+------------+
                       |
                 Tweet Attention
                       |
                Daily Tweet Vector
                       |
                    Daily GRU
                       |
              Temporal Attention
                       |
                Text representation
```

The model learns:

> Which tweets mattered today?

and:

> Which previous days of text mattered?

---

# 12. Text Encoder Choice

## First: reproduce MAN-SF

Use:

**Universal Sentence Encoder (USE)**

- 512-dimensional embedding
- This matches the published MAN-SF setup.

This gives us a clean reproduction.

## Then compare

Only after the architecture works:

1. USE
2. FinBERT
3. PyFin-sentiment
4. Other financial-tweet encoders if necessary

Do NOT repeat dozens of embedding dimensions.

The question should become:

> Which encoder works best when tweet-level and temporal attention are implemented correctly?

---

# 13. Sentiment

We can use a financial-tweet sentiment model as an additional feature.

PyFin-sentiment is worth testing because research specifically evaluated financial-tweet sentiment and reported improvements over VADER, FinBERT and TwitterRoBERTa on its sentiment dataset.

But:

> Sentiment should be an additional signal, not the entire text representation.

The main text representation should still preserve the actual tweet semantics.

---

# 14. Fundamental Encoder

Start simple.

```text
Fundamental features
        |
Normalization
        |
Missingness mask
        |
Small MLP
        |
Fundamental embedding
```

Do not immediately build a fundamental graph.

First establish whether the cleaned fundamental features actually improve the baseline.

---

# 15. Fusion

Do NOT simply concatenate:

```text
price + fundamentals + text
```

Use:

## Bilinear fusion

Why?

Published MAN-SF experiments showed:

```text
Concatenation → MCC ~0.156
Attention fusion → MCC ~0.173
Bilinear fusion → MCC ~0.195
```

Conceptually, bilinear fusion asks:

> What does this text signal mean given the current price situation?

instead of treating both as independent blocks.

---

# 16. Graph

Start with:

## GAT — Graph Attention Network

Do not start with HGT.

GAT is directly supported by StockNet literature and is simpler to reproduce and debug.

Use:

**PyTorch Geometric**

with:

```text
GATConv
```

---

# 17. First Graph

Start simple:

```text
Company
   |
   +-- same sector
   |
Company
```

For example:

```text
AAPL
MSFT
GOOG
NVDA
```

in a technology graph.

Then gradually add:

1. Sector relationships
2. Industry relationships
3. Wikidata company relationships
4. Fundamental similarity
5. Event relationships

Do not add all graph types at once.

---

# 18. Later: Heterogeneous Graph Transformer

Once GAT works, test:

```text
HGTConv
```

using PyTorch Geometric.

Then we can represent:

### Nodes

```text
Company
Sector
Event
Fundamental state
```

### Edges

```text
competitor
supplier
customer
same_sector
same_industry
affected_by
announced
```

This is a later improvement, not the starting point.

---

# 19. Events

Events are worth adding because prior research shows that fine-grained financial events can improve stock-movement prediction.

Start simple.

Classify tweets/news into:

```text
Earnings
Product launch
M&A
Lawsuit
CEO change
Regulation
Dividend
Analyst upgrade
Analyst downgrade
Guidance
```

Convert them into embeddings/features.

Do NOT introduce a huge LLM event pipeline immediately.

Later, if event information proves useful, investigate LLM-based event extraction.

---

# 20. Prediction Head

First:

```text
Stock representation
       |
       v
Binary classifier
       |
       v
UP / DOWN
```

Output:

```text
P(UP)
P(DOWN)
```

Later add multi-task prediction:

```text
                    Shared representation
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
         Direction       Return       Volatility
```

This should come only after the classification model is strong.

---

# 21. Market Regime

Later add an HMM-based market regime.

Possible regimes:

```text
Bull / low volatility
Bull / high volatility
Bear / low volatility
Bear / high volatility
Sideways
```

Use the regime as another feature.

Do not make this part of the first reproduction.

---

# 22. Exact Experiment Sequence

This is the most important section.

## Experiment 0 — Current baseline

Already completed.

Current best:

```text
MCC ≈ 0.046
AUC ≈ 0.530
```

---

## Experiment 1 — Exact StockNet baseline

Implement:

```text
StockNet labels
StockNet split
Trading-day alignment
5-day window
Price + tweets
```

Goal:

Establish a trustworthy benchmark.

---

## Experiment 2 — Reproduce MAN-SF

Implement:

```text
Price:
GRU + temporal attention

Tweets:
USE 512
+ tweet-level GRU
+ tweet attention
+ daily GRU
+ temporal attention

Fusion:
Bilinear fusion

Graph:
GAT
```

Start from the published hyperparameters:

```text
USE = 512
GRU hidden = 64
GAT heads = 8
GAT layers = 2
Adam
learning rate = 5e-4
early stopping on validation MCC
```

Target approximately:

```text
Accuracy ≈ 60.8%
F1 ≈ 60.5%
MCC ≈ 0.195
```

If we cannot reproduce this approximately:

> STOP and debug.

Do not add more models.

---

# 23. Experiment 3 — Add Fundamentals

Take the reproduced MAN-SF model.

Add:

```text
SEC fundamentals
+
missingness masks
+
days since filing
+
fundamental changes
```

Then compare:

```text
MAN-SF
vs
MAN-SF + Fundamentals
```

This becomes our first major extension.

---

# 24. Experiment 4 — Compare Text Encoders

Keep everything else identical.

Test:

```text
USE
vs
FinBERT
vs
PyFin-sentiment
```

Important:

The architecture must stay:

```text
tweet-level attention
+
daily aggregation
+
temporal attention
```

Otherwise the comparison isn't clean.

---

# 25. Experiment 5 — Events

Add:

```text
Events
```

Compare:

```text
Model without events
vs
Model with events
```

This tells us whether events actually contribute.

---

# 26. Experiment 6 — Graph Improvements

Test sequentially:

```text
No graph
       |
       v
Sector GAT
       |
       v
Sector + industry GAT
       |
       v
Company relationship GAT
       |
       v
Fundamental similarity GAT
```

Record MCC/F1 after every step.

---

# 27. Experiment 7 — HGT

Only if GAT has demonstrated value:

```text
GAT
vs
HGT
```

HGT becomes useful when we have multiple node/edge types.

---

# 28. Experiment 8 — Market Regime

Add:

```text
HMM regime
```

Compare:

```text
Model without regime
vs
Model with regime
```

---

# 29. Experiment 9 — Multi-task prediction

Add:

```text
Direction
+
Return
+
Volatility
```

Then investigate whether the extra tasks improve directional prediction.

---

# 30. Experiment 10 — Trading evaluation

Only after the statistical model is stable.

Backtest:

```text
P(UP) > threshold
      |
      v
BUY
```

and compare against:

- Buy and hold
- Market index
- Simple momentum strategy

Include:

```text
transaction costs
slippage
turnover
```

---

# 31. Tools and Frameworks

Use one consistent stack.

## Data

```text
Python
Pandas
NumPy
PyArrow / Parquet
```

## Deep learning

```text
PyTorch
```

## Text

```text
TensorFlow Hub
  └── USE 512
```

for MAN-SF reproduction.

Later:

```text
Hugging Face Transformers
  └── FinBERT
```

## Graph

```text
PyTorch Geometric
  ├── GATConv
  └── HGTConv
```

## Evaluation

```text
scikit-learn
```

for:

```text
Accuracy
F1
MCC
ROC-AUC
Precision
Recall
```

## Statistics

```text
SciPy
```

for:

```text
confidence intervals
significance tests
```

## Explainability

```text
SHAP
attention weights
GAT attention weights
```

---

# 32. What We Are NOT Doing

Do not randomly add:

```text
FinBERT 768
FinBERT 128
FinBERT 64
VADER
RoBERTa
DeBERTa
Qwen
Llama
CNN
LSTM
Transformer
GRU
GAT
HGT
GAN
Federated Learning
```

all at once.

Do not randomly tune hundreds of technical indicators.

Do not optimize thresholds on the test set.

Do not randomly shuffle time-series data.

Do not use future fundamentals.

Do not zero-fill missing fundamentals without a missingness indicator.

Do not compare scores across papers unless the datasets, labels and splits are comparable.

---

# 33. Final Target Architecture

After the incremental experiments, the final candidate is:

```text
                         STOCK-DAY t
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
              PRICE       FUNDAMENTALS    TWEETS
                |             |             |
              Bi-GRU          MLP        Tweet Encoder
                |             |             |
          Temp Attention      |        Tweet Attention
                |             |             |
                |             |        Daily Tweet Vector
                |             |             |
                |             |          Daily GRU
                |             |             |
                |             |       Temporal Attention
                |             |             |
                +-------------+-------------+
                              |
                     BILINEAR FUSION
                              |
                    COMPANY REPRESENTATION
                              |
                     COMPANY RELATION GRAPH
                              |
                         MULTI-HEAD GAT
                              |
                       MARKET REGIME
                              |
                   +----------+----------+
                   |                     |
                   v                     v
              Direction               Return
                   |                     |
                 UP/DOWN             Expected %
```

Later:

```text
HGT
+
Events
+
Volatility prediction
+
Regime conditioning
```

---

# 34. Research Story

The research contribution should NOT be:

> "We tried a lot of ML models."

Instead:

> **We investigate whether integrating point-in-time fundamentals with social-media information and inter-stock relationships improves next-day stock movement prediction over established StockNet architectures.**

The experiments should answer:

### Question 1

Does hierarchical tweet attention outperform simple mean pooling?

### Question 2

Do point-in-time fundamentals improve StockNet-style prediction?

### Question 3

Does bilinear fusion outperform simple concatenation?

### Question 4

Do inter-stock relationships improve prediction?

### Question 5

Do financial events add information beyond sentiment?

### Question 6

Does HGT improve over GAT once heterogeneous relationships are introduced?

### Question 7

Does market-regime information improve robustness?

That is a much stronger research story.

---

# 35. Priority Order

If time is limited, do exactly this:

```text
PRIORITY 1
Fix StockNet split + labels + trading-day alignment
        |
        v
PRIORITY 2
Reproduce MAN-SF
        |
        v
PRIORITY 3
Add SEC fundamentals properly
        |
        v
PRIORITY 4
Fix tweet aggregation
        |
        v
PRIORITY 5
Bilinear fusion
        |
        v
PRIORITY 6
GAT
        |
        v
PRIORITY 7
Events
        |
        v
PRIORITY 8
HGT
        |
        v
PRIORITY 9
Market regime
        |
        v
PRIORITY 10
Multi-task + trading optimization
```

## Most important immediate task

**Do Experiment 1 and Experiment 2 first.**

Do not touch FinBERT again yet.

Our immediate goal is:

> **Get the exact StockNet benchmark working, then reproduce MAN-SF's ~0.195 MCC.**

Once that works, we have a solid foundation. Then every improvement we make has a clear scientific reason and a measurable effect.
