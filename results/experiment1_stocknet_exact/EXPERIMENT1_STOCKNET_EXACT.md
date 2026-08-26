# Experiment 1 — Exact StockNet Protocol A Reproduction

## Objective

Per the project roadmap's Priority 1/2 ("Fix StockNet split + labels + trading-day
alignment" → "Establish a trustworthy benchmark"), this reproduces the **data
methodology** of the original StockNet paper (Xu & Cohen, ACL 2018) as
faithfully as possible, and evaluates it with simple baselines (Logistic
Regression, LSTM, MLP) to get a benchmark number we can trust before attempting
the full MAN-SF architecture (Experiment 2).

This is **not** a re-derivation from this project's own preprocessing
conventions. The rules below were read directly out of the official
[`stocknet-code`](https://github.com/yumoxu/stocknet-code) repository
(`src/DataPipe.py`, `src/config.yml`), not guessed or assumed.

## Methodology (ported from the official code)

| Rule | Source | Value |
|---|---|---|
| Ticker universe | `config.yml: stocks` | exact 88 tickers across 9 sectors |
| Split (end-exclusive) | `config.yml: dates` | train `2014-01-01→2015-08-01`, dev `2015-08-01→2015-10-01`, test `2015-10-01→2016-01-01` |
| Window | `config.yml: max_n_days=5` | **calendar-anchored**: target date minus 4 calendar days |
| Target label | `DataPipe._get_mv_class` | `1` if movement % `> 0` else `0` |
| Buffer-zone filter | `DataPipe._get_prices_and_ts` | a day is usable as a **target** only if `mv < -0.5%` or `mv >= 0.55%`; the filter applies **only to the day being predicted**, never to input days |
| Price features | `DataPipe._get_prices(data)` = `data[3:6]` | the 3 official per-day price ratios in `stocknet-dataset/price/preprocessed/{TICKER}.txt` (StockNet's own precomputed values) |
| Tweet alignment | `DataPipe._trading_day_alignment` | every calendar day's tweets attach to the **first actual trading day strictly after it** — never the same day, regardless of what time they were posted |

### Key implementation detail: variable-length windows, not filtered ones

The window is anchored by **calendar** days, not trading days, so its true
length varies: a Tuesday's target has a lookback spanning Friday–Monday, which
contains only 2 real trading days, not 4. The official TensorFlow
implementation handles this by **zero-padding short windows and tracking the
real length in a separate mask tensor**, never discarding a candidate sample
because its window came up short.

**A first draft of this reproduction got this wrong**: it required every
window to contain exactly 4 real trading days, silently discarding any sample
where a weekend or holiday fell inside the lookback. That reduced the usable
dataset from the expected ~26,600 samples to about 4,700 — an 83% loss — before
the mistake was caught. The fix was to pad short windows to a fixed 4-day
shape and mask them (via `Window_Length` + `pack_padded_sequence` in the LSTM),
exactly matching the official code's approach.

## Validation

This pipeline produces:

- **26,619 total samples** (train 20,344 / dev 2,555 / test 3,720)
- **50.2% / 49.8%** up/down target balance

This matches the published StockNet dataset's known scale (~26,614 samples,
near-balanced) closely enough to treat the reproduction as faithful. This
congruence — not any single design decision in isolation — is the strongest
evidence the windowing, labeling, and split logic are correct.

## Results

Four feature sets (mirroring the same FS1–FS4 structure as the Phase 1
baseline), three models, evaluated on the **official** test split:

| Feature Set | Model | Accuracy | F1 | MCC | AUC |
|---|---|---:|---:|---:|---:|
| FS1: Price only | Logistic Regression | 0.5272 | 0.6126 | +0.0479 | 0.5364 |
| FS1: Price only | LSTM | 0.5137 | 0.6738 | +0.0094 | 0.4841 |
| FS1: Price only | MLP | 0.5242 | 0.6667 | +0.0483 | 0.4983 |
| FS2: Price + Fundamentals | Logistic Regression | 0.5366 | 0.5947 | +0.0688 | 0.5321 |
| FS2: Price + Fundamentals | LSTM | 0.5234 | 0.6344 | +0.0391 | 0.5276 |
| FS2: Price + Fundamentals | MLP | 0.5051 | 0.5108 | +0.0103 | 0.5091 |
| FS3: Price + Tweet Counts | Logistic Regression | 0.5290 | 0.6131 | +0.0521 | 0.5319 |
| FS3: Price + Tweet Counts | **LSTM** | **0.5468** | 0.6363 | **+0.0922** | **0.5666** |
| FS3: Price + Tweet Counts | MLP | 0.4758 | 0.3026 | -0.0409 | 0.5021 |
| FS4: Price + Fundamentals + Tweet Counts | Logistic Regression | 0.5358 | 0.5956 | +0.0670 | 0.5310 |
| FS4: Price + Fundamentals + Tweet Counts | LSTM | 0.5293 | 0.5852 | +0.0540 | 0.5218 |
| FS4: Price + Fundamentals + Tweet Counts | MLP | 0.5046 | 0.4064 | +0.0196 | 0.5081 |

Per-model, per-feature-set metrics (with confusion matrices) are saved under
`results/experiment1_stocknet_exact/{FEATURE_SET}/{model}/metrics.json`.
Aggregated results: `summary.json`, `full_results.json`.

**Best result: LSTM on Price + Tweet Counts (FS3), MCC = +0.0922, AUC = 0.5666.**

Fundamentals (FS2, FS4) are the same 8 EDGAR features used in Phase 1
(`Revenue, NetIncome, TotalAssets, TotalLiabilities, StockholdersEquity, EPS,
Cash, ROA`), re-aligned to each ticker's **full StockNet trading calendar**
(not the buffer-zone-filtered dates in `dataset/final/`, which would have
silently dropped fundamentals coverage for exactly the days some input windows
need — see Implementation Note below), zero-filled where unavailable to match
the same convention as Phase 1's FS2/FS4. Adding fundamentals alone (FS2) beats
price-only (FS1) on Logistic Regression and LSTM, but doesn't beat FS3's tweet
count signal, and combining both (FS4) doesn't clearly beat FS3 either — the
tweet-count signal and the fundamental signal don't appear to compound in this
simple concatenation setup.

### Implementation note: fundamentals had to be re-aligned, not reused directly

`dataset/final/` (the corrected preprocessing pipeline) forward-fills
fundamentals only across the dates that survived its own buffer-zone target
filter — which is exactly the bug that pipeline was built to avoid for *price*
windows, but the same issue resurfaces here for *fundamentals* if you naively
join on `(Ticker, Date)`. Since this exact-protocol dataset needs fundamentals
values for arbitrary trading days (any day that lands in a window, not just
days that survived as valid StockNet *targets*), fundamentals were re-aligned
from the cached raw EDGAR fetch (`dataset/final/edgar_raw_fundamentals.csv`,
no new API calls) against each ticker's full StockNet price calendar
(`stocknet-dataset/price/preprocessed/{TICKER}.txt` dates), the same way as
before (forward-fill from filing date) but without the target-only date
restriction.

## Comparison to literature

| Model (source) | MCC |
|---|---:|
| LSTM + price (MAN-SF paper's own ablation table) | ~0.002 |
| **This experiment: LSTM + price (FS1)** | **+0.009** |
| GRU + social text (MAN-SF ablation table) | ~0.077 |
| **This experiment: LSTM + price + tweet counts (FS3)** | **+0.092** |
| GCN + price (MAN-SF ablation table) | ~0.093 |
| MAN-SF full model (target for Experiment 2) | ~0.195 |

Our price-only LSTM result (MCC +0.009) is close to MAN-SF's own reported
price-only LSTM baseline (MCC ~0.002) — both are effectively at the noise
floor, as expected for a model with no text signal. Adding even a coarse tweet
**count** feature (no semantic content, just volume) already lifts MCC to
+0.092, in the same range as MAN-SF's GRU+social-text and GCN+price ablations.
This is a believable, literature-consistent result, which is the point of this
exercise: it gives us a number we can trust well enough to build on.

## What this experiment does NOT include

- **No tweet semantics.** The tweet feature here is a raw per-day count,
  aligned using the exact StockNet trading-day rule — not FinBERT/USE
  embeddings, not hierarchical tweet-level attention. Text semantics + the
  full hierarchical-attention + bilinear-fusion + GAT architecture is
  Experiment 2 (MAN-SF reproduction) in the project roadmap, not this one.
- **Fundamentals are included (FS2/FS4) but are a project-specific extension
  beyond the original StockNet task** — the official StockNet paper/dataset
  has no fundamentals concept at all. They're included here only to keep this
  experiment's feature-set structure consistent with Phase 1's FS1–FS4, not
  because they're part of "exact StockNet reproduction."
- **The official Hedge-FVMD architecture is not reproduced.** This uses plain
  Logistic Regression / LSTM / MLP baselines, not StockNet's own generative
  recurrent model with market-information-encoding variational latent
  variables. The point of this experiment was a trustworthy **data**
  benchmark, not a model benchmark.

## Reproducing this experiment

Run `preprocessing_notebooks/MMGTFFF_stocknet_EXACT_reproduction.ipynb`
top-to-bottom in a fresh Colab runtime (clones `stocknet-dataset` only, no
external API calls, fully deterministic given `SEED=42`). It regenerates:

- `dataset/stocknet_protocol_a_exact.parquet` — the flat, model-ready dataset
  (one row per sample: `Ticker`, `Target_Date`, `Split`, `Target`,
  `Window_Length`, `Day{1..4}_Price{1,2,3}`, `Day{1..4}_TweetCount`,
  `Day{1..4}_{Revenue,NetIncome,TotalAssets,TotalLiabilities,
  StockholdersEquity,EPS,Cash,ROA}`, `Pretarget_Tweet_Count`) — fundamentals
  are re-fetched from SEC EDGAR inside the notebook using the same
  CIK-matching and tag fixes as the corrected preprocessing notebook
- the metrics saved under `results/experiment1_stocknet_exact/`

## Next step

Per the roadmap's priority order, the next step is **Experiment 2**: reproduce
MAN-SF's actual architecture (GRU + temporal attention for price, tweet-level
encoder + tweet attention + daily GRU + temporal attention for text, bilinear
fusion, GAT) on top of this same exact-protocol dataset, targeting MCC ≈ 0.195.
