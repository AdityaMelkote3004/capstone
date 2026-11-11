# Experiment 11 — results (the paper's numbers)

Test set: StockNet / ACL18 (Adv-ALSTM preprocessing), 2015-10-01 .. 2015-12-31,
n = 3,720 predictions on 64 trading days. Verified identical, row by row
(stock, date, label), to the test set of Experiments 1-10.

**Revision after external review (current numbers).** Baselines were re-tuned on
validation (34 settings, see PREREGISTRATION.md), every final model was rerun with
all test predictions saved (`preds/`, tags `R_*` and `FEDlong`), and three analyses
were added: a day-block bootstrap, a trading simulation and a walk-forward
evaluation. Numbers come from `paper/revision_tables.py` (tables, macro-F1 from
saved predictions), `paper/revision_analysis.py` (pooled-MCC bootstrap, trading)
and `paper/revision_extra.py` (within-day bootstrap, buy-and-hold, per-period gains);
CSVs `rev_*.csv`. Mean ± s.d. over ten seeds unless marked.

## Main results (standard test period)

| Model | Pooled MCC | Acc | Macro-F1 | AUC | Within-day | Day-collapse | Day-level p vs MMGT |
|---|---|---|---|---|---|---|---|
| Always predict "up" | 0 | 0.513 | 0.339 | 0.500 | – | – | – |
| Logistic regression, C = 0.01 (1 run) | 0.027 | 0.515 | 0.513 | 0.513 | −0.012 | 0.061 | 0.024 |
| LSTM (tuned) | 0.055 ± 0.026 | 0.519 | 0.489 | 0.535 | **0.025** | 0.082 | 0.059 |
| ALSTM (tuned) | 0.041 ± 0.031 | 0.519 | 0.468 | 0.530 | 0.018 | 0.022 | 0.011 |
| MLP (tuned) | 0.057 ± 0.020 | 0.528 | 0.525 | 0.536 | 0.000 | 0.110 | 0.075 |
| Tweet-count LSTM, Exp 1 (5 seeds) | 0.002 ± 0.011 | – | – | – | – | – | – |
| RGCN graph, Exp 6 (5 seeds; graph possibly optimistic) | 0.060 ± 0.018 | 0.527 | 0.494 | 0.541 | 0.005 | – | – |
| MMGT-base (price only) | 0.139 ± 0.049 | 0.558 | 0.534 | 0.593 | 0.008 | 0.130 | 0.82 |
| Profile only (no ranking term) | 0.137 ± 0.033 | 0.567 | 0.560 | 0.599 | 0.031 | 0.131 | 0.74 |
| **MMGT** | **0.146 ± 0.029** | **0.570** | **0.565** | **0.602** | 0.023 | **0.219** | – |
| MMGT federated (5 seeds) | 0.131 ± 0.062 | 0.566 | 0.553 | 0.598 | −0.004 | 0.140 | 0.66 |

- Run-level Welch p, MMGT vs each tuned baseline: all < 1e-6. All ten MMGT runs
  (lowest 0.097) exceed the best baseline mean (0.060).
- Day-block bootstrap (64 days resampled, 2,000 times) of MMGT minus model:
  MLP +0.089 [−0.010, 0.172], LSTM +0.091 [−0.004, 0.174], ALSTM +0.105
  [0.025, 0.183], LR +0.119 [0.020, 0.211]. Significant at the day level against
  ALSTM and LR only.

## The story: market timing vs stock selection

- Per-stock models select stocks but time the market weakly and unstably
  (LSTM within-day 0.025, day-collapse 0.082 ± 0.108).
- MMGT-base times the market (day-collapse 0.130 of pooled 0.139) with near-zero
  selection (0.008).
- MMGT's gain is mostly market timing (day-collapse 0.219). The profile adds
  selection: within-day 0.008 → 0.031 (8/10 runs, run-level p = 0.011; day-level
  +0.023 [−0.007, 0.051], p = 0.14). MMGT vs MMGT-base within-day +0.015 (day-level
  p = 0.24).
- Across the three test periods, MMGT's within-day gain over MMGT-base has the
  same sign each time (+0.014, +0.010, +0.015); 15 of 20 paired runs positive,
  sign test p = 0.041 (one-sample t p = 0.076).

## Walk-forward (settings from the standard split, not re-tuned)

Period A: test 2015-06-01..07-31 (44 days, 2,293 samples, 48.3% up).
Period B: test 2015-08-03..09-30 (42 days, 2,555 samples, 44.6% up). B is the
standard split's validation period, so it is not fully out-of-sample.
Five seeds in A and B (LR: one deterministic run).

| Model | MCC A | MCC B | MCC Oct–Dec | Avg | WD A | WD B | WD Oct–Dec |
|---|---|---|---|---|---|---|---|
| LR | 0.024 | **0.115** | 0.027 | 0.055 | −0.027 | −0.015 | −0.012 |
| MLP | 0.003 | 0.096 | 0.057 | 0.052 | −0.008 | −0.002 | 0.000 |
| LSTM | 0.029 | 0.022 | 0.055 | 0.035 | 0.002 | −0.008 | **0.025** |
| MMGT-base | **0.135** | 0.111 | 0.139 | **0.128** | 0.023 | 0.007 | 0.008 |
| MMGT | 0.048 | 0.106 | **0.146** | 0.100 | **0.037** | **0.017** | 0.023 |

Pooled-MCC rankings move with market timing (MMGT day-collapse 0.029 / 0.174 / 0.219).

## Trading simulation (standard test period, annualised Sharpe, mean over runs)

Long-short: top-10 minus bottom-10 by predicted probability among all window-valid
stocks. Timing: all stocks long if mean probability > 0.5, else short. Daily
rebalancing, costs per unit turnover. Equal-weight buy-and-hold: Sharpe 1.39,
return +5.0%, max drawdown 6.5%.

| Model | LS 0 bps | LS 10 bps | Timing 0 bps | Timing 10 bps | Timing return |
|---|---|---|---|---|---|
| LR | 0.02 | −6.19 | 1.62 | −0.04 | +5.9% |
| MLP | 0.96 | −4.75 | 2.02 | 0.39 | +7.5% |
| LSTM | 1.59 | −1.64 | 0.96 | 0.49 | +3.5% |
| ALSTM | **3.30** | **0.53** | 0.46 | 0.04 | +1.5% |
| MMGT-base | 0.28 | −2.94 | 2.06 | 1.59 | +7.6% |
| Profile only | −0.13 | −4.25 | 2.69 | 1.76 | +10.0% |
| MMGT | −0.23 | −4.23 | **4.24 ± 1.73** | **3.16** | **+15.9%** |
| Federated | −2.32 | −6.31 | 3.34 | 2.56 | +12.5% |

MMGT's within-day gain does not reach the top/bottom-10 extremes; the ALSTM is the
best stock picker in money terms.

## Ablations

| Variant | Pooled MCC | Within-day | Note |
|---|---|---|---|
| MMGT-base (10) | 0.139 ± 0.049 | 0.008 | |
| − stock attention (5) | 0.059 ± 0.033 | 0.022 | vs base on same seeds 0.130, p = 0.048 |
| 7 of 11 price features (10) | 0.121 ± 0.032 | −0.001 | vs base p = 0.21 |
| + multi-modal profile (10) | 0.137 ± 0.033 | **0.031** | within-day vs base p = 0.011, 8/10 seeds |
| + profile + graph-biased attention (10) | 0.143 ± 0.046 | 0.028 | vs profile model p = 0.45 |
| **MMGT** = + profile + ranking term (10) | **0.146 ± 0.029** | 0.023 | vs profile model: pooled p = 0.46, WD p = 0.35 |
| MMGT − market context (5) | 0.078 ± 0.040 | 0.024 | vs MMGT on same seeds 0.131, p = 0.12 |
| profile + graph-masked attention | — | — | val 0.223 vs 0.279; not taken to test (pre-registered rule) |

## Federated (FedAvg, nine sector clients, profile model without ranking term)

Long schedule (tag `FEDlong`): up to 150 rounds, patience 30, seeds 0-4. Best
rounds 0, 92, 69, 58, 83 (seed 0 never improved after round 1).

| Training (same five seeds) | Pooled MCC | Within-day | Day-collapse | Acc | AUC |
|---|---|---|---|---|---|
| Centralised | 0.115 | 0.024 | 0.103 | 0.557 | 0.587 |
| Federated | 0.131 ± 0.062 | −0.004 | 0.140 | 0.566 | 0.598 |
| paired p | 0.67 | **0.037** | 0.52 | 0.63 | 0.66 |

Pre-registered rule (MCC ≥ 0.10 and not significantly below centralised): **met,
federated stays in the paper.** Federation keeps the score and the market timing
but loses stock selection (day-level p vs MMGT 0.069). Earlier short-schedule run
(`FEDc`, 40 rounds, patience 10): 0.040, matching a one-epoch centralised control;
superseded.

## Correction: target-day leak (found during review, before publication)

The first version of the stock profile was keyed to the target date in the
modeling parquet, which contains only labelled rows; row presence on day t
revealed whether that stock's move on t left the buffer zone (26,584 / 16,782
cells, zero exceptions). Results tagged `F1_peer`, `F2_ours`, `F3_nomc`,
`G2_bias4` and `FED_fedavg` used that version and are **void**; they remain in
the CSVs only as the record. The corrected profile (`src/data/acl18_extra.py`,
`build_profile_v2`) passes `tests/test_profile_leakage.py`; results tagged
`F1c`, `F2c`, `F3c`, `G2c`, `FEDc`, `R_*` and `FEDlong` use it.
`F0_dtml`, `F4_nosa`, `S4_ret_trend` and all baselines never used the profile.

## Search effort (all on validation)
48 grid runs (DTML's search space) + 12 (stage C) + 18 (stage D) + 9 (stage E)
+ 34 baseline settings (revision).
