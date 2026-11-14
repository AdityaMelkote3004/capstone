# Supplementary material

*Beyond Pooled Scores: Separating Market Timing and Stock Selection in Multimodal Stock Forecasting*

Every number here is produced by a script in this repository from saved results. The
script and its output file are named in each section.

---

## S1. Protocol: what was confirmatory, what was not

The protocol log is `results/experiment11_dtml/PREREGISTRATION.md`. Its git history
timestamps every entry, and each test-period run came after the entry that fixed it.

| Date (commit) | Entry | Status |
|---|---|---|
| 2026-11-09 (c7e7e33) | Exp 11: MMGT development search on validation only; test ladder fixed | confirmatory for MMGT |
| 2026-11-09/10 (89e23cf, 367699f, a87f055) | Feature subsets, graph-biased attention, federated training pre-registered | confirmatory ablations |
| 2026-11-10 (e2d20a7) | Target-day leak found in the profile; all profile results voided and rerun | correction |
| 2026-11-11 (d10be00, 0e9b70c) | Revision plan, federated keep/remove rule, tuned-baseline selection recorded before test | confirmatory |
| 2026-11-11 (ad6a2ba, dc4ffb6, 2ba1a02) | Exp 12, MMGT-D: pre-registered, selected on validation, failed its guard | confirmatory (negative) |
| 2026-11-13 (38402a5, 1a61496, b2ab254) | Exp 12b, MMGT-D2: pre-registered, selected on validation, met its endpoint | pre-registered, but **designed after MMGT's test results were known**; not an independent confirmation |
| 2026-11-14 (91621de, d2ed258) | Exp 13: ablation of the unselected MMGT-D2 variants (post-hoc, disclosed) and the market-information probe (rule fixed before running) | post-hoc / pre-specified analysis |
| 2026-11-14 (1116d18) | Round 4: period B quarantined as in-sample | re-analysis, no new training |
| 2026-11-14 (b767482) | Round 5: block and hierarchical bootstrap, multiplicity rules fixed before computation | re-analysis, no new training |

Earlier project experiments (Exp 1–10) were exploratory. This includes the tweet-embedding
experiments behind the use of tweet volume rather than tweet content. They are summarised
in `results/CAPSTONE_FINDINGS.md` and are not part of the paper's evidence.

---

## S2. Implementation

### Data and labels
- **Data:** StockNet [Xu and Cohen, 2018] with the price features released by Adv-ALSTM
  [Feng et al., 2019].
  - 87 stocks, on a shared trading calendar from 2013-06-03 to 2015-12-31.
  - The days before 2014-01-02 are used only as input history.
- **Labels:** taken from StockNet unchanged.
  - Up: adjusted-close return greater than +0.55%.
  - Down: return below −0.5%.
  - Days in between are an unlabelled buffer.
  - These thresholds are StockNet's own. They were not tuned here.
- **Splits (standard):**

  | Split | Dates |
  |---|---|
  | Train | 2014-01-02 to 2015-07-31 |
  | Validation | 2015-08-03 to 2015-09-30 |
  | Test | 2015-10-01 to 2015-12-31 (64 days, 3,720 labelled stock-days) |

- **Walk-forward periods (same frozen settings):**

  | Period | Train from | Validation | Test |
  |---|---|---|---|
  | A | 2014-01-02 | 2015-04-01 to 2015-05-29 | 2015-06-01 to 2015-07-31 |
  | B | 2014-01-02 | 2015-06-01 to 2015-07-31 | 2015-08-03 to 2015-09-30 |

  Period B is the standard validation period, so it is in-sample.
  (`src/data/acl18.py`)

### Inputs for the prediction on day *t* (all end at the close of *t*−1)
- **Price window:** 10 days × 11 features.
  - Open, high and low relative to close.
  - Close and adjusted-close changes.
  - Price relative to its 5-, 10-, 15-, 20-, 25- and 30-day moving averages.
- **Market summary:**
  - For each day, the equal-weighted cross-sectional mean of the same 11 features,
    taken over the stocks that have data that day.
  - The model reads the last 10 days of this series (a 10 × 11 window), identical for
    every stock on a day.
  - This replaces the S&P 500 index used by DTML [Yoo et al., 2021], which is not in
    the released data.
- **Profile (47 features).** Standardised with training rows only
  (`src/data/acl18_extra.py`, `src/data/node_features.py`).
  - *Volatility (6):*
    - realised volatility over 5, 10 and 20 days;
    - EWMA volatility (λ = 0.94);
    - the previous day's return and its absolute value.
  - *Fundamentals (20):* 13 SEC EDGAR variables, each dated by its filing date and first
    used the day after filing:
    - Revenue, NetIncome, TotalAssets, TotalLiabilities, StockholdersEquity, EPS, Cash
      (the six level variables other than EPS are log-transformed);
    - Revenue_Growth, NetIncome_Growth, EPS_Growth, Profit_Margin, Debt_To_Equity, ROA;
    - plus 7 missing-value flags.
  - *Tweet volume (5):* log(1 + count) for each of the 4 previous days, and the total.
  - *Peer features (16):*
    - the neighbour mean of 3 prior-day quantities (return, absolute return, EWMA
      volatility);
    - the stock's own value minus that mean (3);
    - the log neighbour count (1);
    - a sector one-hot (9).
- **Stock graph (neighbours for the peer features).** The union of three relations, all
  built from information before *t* (`src/data/graph_edges.py`):
  - *sector:* static same-sector pairs;
  - *correlation:* each stock linked to its 5 partners with the largest |Pearson ρ| of
    daily returns over the previous 60 trading days (at least 10 observations);
  - *co-mention:* pairs whose cashtags appear in the same tweet during the 5 calendar
    days before *t*.
- **No look-ahead:** for each of the 504 target days, all data from that day onward was
  deleted and the features and graph were rebuilt; they did not change
  (`tests/test_profile_leakage.py`).

### MMGT architecture (`src/models/dtml.py`; hidden size *d* = 64)
1. **Stock encoder:** Linear(11→64), then an LSTM(64), then temporal attention over the
   10 days, weighted by relevance to the last day.
2. **Normalisation:** across the day's valid stocks, with a learned scale and shift.
3. **Profile:** Linear(47→64), added to the encoding.
4. **Market summary:** an encoder of the same form applied to the 10 × 11 market window,
   added with weight β = 1.
5. **Stock attention:**
   - multi-head self-attention (4 heads, d = 64, dropout 0.15);
   - the softmax runs over the stocks with a valid window (a key-padding mask);
   - residual connection, then tanh(z + MLP(z)) with an MLP of 64→256→64, then
     LayerNorm.
6. **Output:** Linear(64→1), one logit per stock.

About 132k parameters.

### MMGT-D2 (`src/models/dtml_disentangled.py`)
- **Timing branch:** a full MMGT. *m_t* is the mean of its logits over the valid stocks.
- **Selection branch:**
  - the same encoder, profile and attention, but no market-summary encoder;
  - its states are centred over the day's stocks before a bias-free Linear(64→1), so
    the excess logits *e* sum to zero each day.
  - The branch can still pick up market-wide information through attention across all
    stocks; the probe confirms that it does.
- **Adversary:**
  - an MLP(128→64→1) on [mean |z|, s.d. z] of the centred states, passed through
    gradient reversal (weight 1);
  - it is trained to predict whether the mean target-day return of the valid stocks is
    positive;
  - its loss weight is λ_adv = 0.1.
- About 273k parameters.

### Training (both models)
- **Batches:** one trading day per mini-batch, containing all of that day's stocks.
- **MMGT loss:** BCE + 1.0 × ‖output layer‖² + α × RankNet, with α = 1. RankNet here is
  the mean softplus of (down score − up score) over that day's up/down pairs.
- **MMGT-D2 loss:**
  - BCE on the final logit;
  - λ_aux × (BCE + RankNet) on the timing logits *l*, with λ_aux = 1;
  - α × RankNet on *e*;
  - λ_adv × the adversary's BCE.
- **Optimisation:** Adam, learning rate 1e-4, gradient-norm clip 1.0, at most 200 epochs.
- **Model selection:** early stopping with patience 30 on validation MCC; the checkpoint
  with the best validation MCC is kept.
- **Runs:** seeds 0–9 (`src/utils/seed.py`). Training ran on CPU only (PyTorch 2.8).
  Walk-forward periods and some ablations use 5 seeds because of this compute budget.

---

## S3. Evaluation definitions and characterisation

**MCC.** Computed with `sklearn.metrics.matthews_corrcoef`. When the denominator is zero
(for example, constant predictions), MCC is defined as 0, following sklearn's convention.
Macro-F1 is the macro-averaged F1 score: the mean of the up-class and down-class F1.

**Within-day MCC** (`src/utils/metrics.py`, `paper/rev5_stats.py` → `rev5_stats.json`).
- **Population:** the labelled stocks *S_t* of each day.
- **Threshold:** the median predicted probability over *S_t*. A stock predicted at or
  above the median counts as up.
- **Days scored:** days with only one class are skipped, which leaves T′ = 63 of 64 test
  days.
- **Stocks per day:** 20–83 labelled stocks (median 57).
- **Ties:** no day has tied predictions at the median, for any model.
- **Imbalanced days:**
  - On 32 of 64 days, more than 80% of labels point the same way.
  - The median split always predicts half up and half down, so on such days even a
    perfect ranking cannot reach an MCC of 1.
  - This makes the score conservative, not biased toward any model.
- **Per-day spread:** the run-averaged per-day value is positive on 49–51% of days for
  every model. The quartiles are about [−0.06, 0.00, 0.09], so the averages in the paper
  are small means of noisy daily values.

**Variants** (run means; not used for claims):

| Model | Within-day (median split) | Top vs bottom tercile | All valid stocks, label = sign of return | Rank IC |
|---|---|---|---|---|
| ALSTM | 0.018 | 0.028 | 0.004 | 0.027 |
| LSTM | 0.025 | 0.033 | 0.005 | 0.020 |
| MMGT-base | 0.008 | 0.014 | 0.014 | 0.015 |
| MMGT | 0.023 | 0.029 | 0.001 | 0.001 |
| MMGT-D2 | 0.031 | 0.041 | 0.010 | 0.027 |

- **Tercile variant:** keeps the ordering of the main diagnostic.
- **All-stock variant:**
  - It adds the buffer-zone stocks, whose moves are within about ±0.5%, and uses the
    sign of the close-to-close return as the label.
  - This compresses every model to near zero and changes the ordering.
  - Within-day MCC therefore inherits StockNet's choice to exclude small moves.
  - Rank IC uses all stocks and their realised returns, so the two measures are
    complementary rather than interchangeable.

**Day-collapse MCC** is pooled over labelled stock-days, so days with more labelled
stocks weigh more. It thresholds the day's mean probability at 0.5, so it depends on
calibration.

**Daily market-direction evaluation** (one observation per test day):
- *Score:* the mean predicted probability over the valid stocks.
- *Target:* whether the mean return of the valid stocks is positive.
- *Main metric:* AUC, because it does not depend on a threshold.
- *Intervals:* 95% hierarchical block bootstrap.

| Model | Daily AUC | 95% interval | Daily MCC at 0.5 |
|---|---|---|---|
| Logistic regression | 0.557 | [0.42, 0.68] | 0.061 |
| MLP | 0.569 | [0.45, 0.69] | 0.115 |
| ALSTM | 0.579 | [0.49, 0.69] | 0.066 |
| LSTM | 0.609 | [0.51, 0.71] | 0.128 |
| MMGT-base | 0.640 | [0.53, 0.76] | 0.223 |
| MMGT | 0.660 | [0.54, 0.77] | 0.262 |
| MMGT-D2 | 0.654 | [0.52, 0.77] | 0.248 |
| Market-index-only LR (reference, added after review) | 0.547 | – | – |

- **About the market-index-only logistic regression:**
  - It was added after review.
  - *Features:* only the 10 × 11 market window, standardised on training rows.
  - *Fitting:* pooled labelled training stock-days; C ∈ {1e-3, …, 10} was chosen on
    validation MCC (C = 0.01).
  - *Evaluation:* the test period, once.
  - *Result:* test pooled MCC 0.046 and daily AUC 0.547. A simple market-only model
    therefore does not reproduce MMGT's timing.

**Counterfactual of removing stock-level differences.** Replacing each prediction with
its day's mean probability gives MMGT a pooled MCC of 0.219, against its actual 0.146.
So the stock-level differences in its predictions add nothing to its pooled score.

**Counterfactual of removing day-level differences** (pre-specified; not interpreted):
- *Method:* each stock is predicted up if it lies above its day's mean probability.
- *Results:* 0.070 for MMGT, 0.043 for MMGT-D2 and 0.034 for the LSTM.
- *Why it is not interpreted:* thresholding at the mean does not force a balanced split,
  so each day's predicted up-share follows the skew of that day's predictions, and some
  day-level information leaks back in.

**Rank IC.**
- *Definition:* the Spearman correlation, each day, between predicted probabilities and
  realised returns over all stocks with a valid window and a return, averaged over days
  [Grinold and Kahn, 2000].
- *Which returns:* close-to-close changes from the released price features. Labels come
  from adjusted closes, so the two do not match exactly. On the test grid, 130 up-labels
  have a close-to-close return of at most +0.55%, and 191 unlabelled stocks fall outside
  the buffer. For the same reason, label-threshold sensitivity by relabelling was not
  attempted.

**Agreement between within-day MCC and Rank IC.**
- *Data:* run means over the 17 out-of-sample (model, period) combinations.
- *Correlation:* Pearson r = 0.84, Spearman 0.82.
- *Interval:* a bootstrap that resamples whole models gives [0.66, 0.92]. This accounts
  for models recurring across periods.
- *Caveat:* this is a correlation of aggregated scores, whereas Rank IC is itself a
  daily Spearman correlation.

**Market-direction oracle (hindsight).**
- *Rule:* each labelled stock is predicted as its day's realised majority label (a tie
  counts as up).
- *Results:* MCC 0.625 on the standard test period, 0.606 in period A, and between 0.46
  and 0.65 in each quarter of 2014–2015 (`paper/rev4_oos.py`).
- It uses realised labels, so it is neither a predictor nor a bound on achievable
  performance.

---

## S4. Statistical robustness (`paper/rev5_stats.py` → `rev5_stats.json`)

**Primary method, fixed before computation:**
- A hierarchical bootstrap with 2,000 replicates. Each replicate resamples each model's
  training runs with replacement, and resamples the 64 test days as a circular
  moving-block bootstrap with 5-day blocks [Künsch, 1989].
- The statistic is the difference of the run-averaged scores.
- Each cell below gives the point estimate, the 95% interval and the two-sided p-value.

| Comparison | Primary (blocks of 5, runs resampled) | Blocks of 1 (i.i.d. days) | Blocks of 10 | Days only, blocks of 5 |
|---|---|---|---|---|
| MMGT − LR, MCC | +0.119 [0.008, 0.224] p=0.040 | p=0.025 | p=0.024 | p=0.033 |
| MMGT − LSTM, MCC | +0.091 [0.009, 0.170] p=0.030 | p=0.067 | p=0.082 | p=0.013 |
| MMGT − ALSTM, MCC | +0.105 [0.024, 0.184] p=0.006 | p=0.022 | p=0.027 | p=0.001 |
| MMGT − MLP, MCC | +0.089 [−0.015, 0.189] p=0.098 | p=0.093 | p=0.090 | p=0.090 |
| MMGT − MMGT-base, MCC | +0.008 [−0.068, 0.075] p=0.80 | 0.87 | 0.84 | 0.82 |
| MMGT − federated, MCC | +0.016 [−0.085, 0.098] p=0.71 | 0.74 | 0.72 | 0.69 |
| MMGT − MMGT-D, MCC | +0.083 [0.009, 0.148] p=0.029 | 0.046 | 0.038 | 0.014 |
| MMGT-D2 − MMGT, MCC | +0.005 [−0.043, 0.060] p=0.88 | 0.86 | 0.85 | 0.79 |
| MMGT-D2 − MMGT, within-day | +0.008 [−0.019, 0.035] p=0.51 | 0.54 | 0.52 | 0.42 |
| MMGT-D2 − MMGT, day-collapse | −0.015 [−0.109, 0.078] p=0.76 | 0.73 | 0.76 | 0.61 |
| **MMGT-D2 − MMGT, Rank IC** | **+0.026 [0.004, 0.051] p=0.027** | **0.071** | 0.031 | 0.0005 |
| MMGT-D2 − MMGT, daily AUC | −0.006 [−0.057, 0.046] p=0.78 | 0.82 | 0.80 | 0.66 |
| MMGT-D2 − LR, MCC | +0.124 [−0.002, 0.250] p=0.054 | 0.021 | 0.056 | 0.050 |
| MMGT-D2 − LSTM, MCC | +0.096 [−0.008, 0.194] p=0.079 | 0.093 | 0.134 | 0.054 |
| MMGT-D2 − ALSTM, MCC | +0.110 [0.010, 0.206] p=0.026 | 0.034 | 0.078 | 0.015 |
| MMGT-D2 − MLP, MCC | +0.094 [−0.022, 0.212] p=0.109 | 0.088 | 0.141 | 0.101 |
| MMGT − MMGT-base, within-day | +0.015 [−0.013, 0.042] p=0.28 | 0.36 | 0.24 | 0.15 |
| Profile − MMGT-base, within-day | +0.023 [−0.006, 0.052] p=0.13 | 0.23 | 0.095 | 0.034 |
| Federated − central (seeds 0–4), within-day | −0.028 [−0.074, 0.018] p=0.22 | 0.30 | 0.18 | 0.10 |
| Federated − central (seeds 0–4), MCC | +0.016 [−0.066, 0.115] p=0.78 | 0.72 | 0.74 | 0.58 |

**Multiplicity:**
- The comparisons of MMGT against the four tuned baselines form one family, adjusted
  with Holm's method under the primary method.

  | Baseline | Holm-adjusted p |
  |---|---|
  | ALSTM | 0.024 |
  | LSTM | 0.090 |
  | Logistic regression | 0.090 |
  | MLP | 0.098 |

- The pre-registered primary endpoint of MMGT-D2 (Rank IC against MMGT) is the single
  confirmatory test.

**"Retains pooled MCC":**
- No non-inferiority margin was fixed in advance, so non-inferiority is not claimed.
- The interval for MMGT-D2 − MMGT pooled MCC is [−0.043, 0.060]. It excludes any loss
  larger than 0.043 MCC.

**Levels with 95% primary intervals:**

| Model | MCC | Within-day | Day-collapse | Rank IC |
|---|---|---|---|---|
| LR | [−0.066, 0.126] | [−0.055, 0.033] | [−0.091, 0.209] | [−0.039, 0.046] |
| MLP | [−0.032, 0.150] | [−0.038, 0.039] | [−0.041, 0.275] | [−0.034, 0.041] |
| LSTM | [−0.001, 0.110] | [−0.002, 0.051] | [−0.013, 0.204] | [−0.010, 0.051] |
| ALSTM | [−0.003, 0.090] | [−0.007, 0.045] | [−0.063, 0.120] | [0.001, 0.050] |
| MMGT-base | [0.024, 0.250] | [−0.015, 0.031] | [−0.031, 0.260] | [−0.005, 0.035] |
| MMGT | [0.068, 0.214] | [−0.003, 0.046] | [0.107, 0.324] | [−0.028, 0.028] |
| MMGT-D2 | [0.047, 0.239] | [0.001, 0.059] | [0.071, 0.320] | [−0.001, 0.056] |

**Probe intervals.** The Experiment 13 probe keeps its pre-registered method: an i.i.d.
day bootstrap of the AUC of the seed-averaged score. The paper's conclusions from the
probe rest on yesterday's direction and the 10-day trend, where every lower bound is at
least 0.92.

---

## S5. Trading simulation assumptions (`paper/revision_analysis.py`, `paper/revision_finance.py`)

- **Timing of trades:**
  - Positions for day *t* are set at the close of *t*−1, at the same prices the inputs
    use. Executing at that close is optimistic.
  - Returns are the close-to-close returns of day *t*.
- **Universe:** all stocks with a valid input window, labelled or not, so no target-day
  information is used.
- **Long–short:** equal weight, long the 10 highest predicted probabilities (1/10 each)
  and short the 10 lowest (−1/10 each). The book is dollar-neutral with gross exposure
  2.
- **Long-only:** the top 10 stocks, at 1/10 each.
- **Timing:** if the mean predicted probability is above 0.5, long every valid stock
  with equal weight; otherwise, short every valid stock with equal weight.
- **Costs:**
  - Turnover is the sum of absolute weight changes from the previous day.
  - The cost is bps × turnover, at 0, 5, 10 and 20 bps.
  - Borrowing costs and market impact are ignored.
- **Statistics:**
  - The Sharpe ratio is the mean daily return over its standard deviation, multiplied
    by √252, computed on 64 daily returns.
  - Results are the mean over training runs.
- These are illustrations of the two abilities, not evidence of a deployable strategy.

---

## S6. Not done, and why

These items need new training or new data. They are listed for a future revision.

- **An untouched later test period or a second stock universe.** No such data is
  available offline: StockNet ends on 2015-12-31.
- **Profile-matched per-stock neural baselines.** These would need the same validation
  tuning as the other baselines.
- **A selection branch without attention across all stocks.**
- **Zero-sum versus unconstrained excess logits,** at matched parameter count.
- **Centralised training with sector-restricted context, against federated training.**
  This would separate the effect of federated averaging from the effect of each client
  seeing only one sector.
- **Label-threshold sensitivity with retraining.** See S3 for why relabelling the saved
  predictions is not valid.
