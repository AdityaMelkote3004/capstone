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
but loses stock selection (run-level paired p = 0.037; day-level vs the centralised
profile model on the same seeds: −0.028 [−0.068, 0.015], p = 0.20). Earlier short-schedule run
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

## Review round 2: Rank IC, full trading statistics, leakage, sensitivity

Scripts: `paper/revision_finance.py` (Rank IC, portfolios), `paper/exp12_eval.py` (Exp 12).

**Rank IC** (daily Spearman of predicted probability vs target-day return over all
window-valid stocks; mean over runs), standard period: ALSTM 0.027, LSTM 0.020,
MMGT-base 0.015, MMGT-D 0.014, Profile 0.013, MLP 0.004, LR 0.002, MMGT 0.001,
Federated -0.024. Period A / B: MMGT 0.023 / 0.045, MMGT-base 0.008 / 0.019, LSTM
0.004 / 0.010, MLP -0.018 / 0.020, LR -0.024 / 0.014, MMGT-D 0.033 / 0.069.
Agreement with within-day MCC: across 21 model x period means r = 0.73 (p = 0.0002);
across seeds within the same model r = 0.75 (n = 125 de-meaned runs).
Labelled-stocks-only Rank IC (standard period): MMGT 0.006, ALSTM 0.039, LSTM 0.039.

**Portfolios** (standard period, Sharpe, mean over runs; buy-and-hold 1.39, max DD 6.5%):
long-only top-10 at 0 bps: ALSTM 2.27, LR 2.11, MLP 1.59, LSTM 1.42, MMGT-base 1.41,
MMGT-D 0.58, MMGT 0.52, Federated 0.24. Best timing portfolio (MMGT) and best
stock-picking portfolios (ALSTM) unchanged at 0/5/10/20 bps. MMGT timing max DD 4.1%.
Full table (Sortino, Calmar, turnover, all costs) in `paper/revision_finance.json`.

**Leakage**: `python tests/test_profile_leakage.py --all` -- all 504 target days, zero
profile difference, all rebuilt graph edges identical. PASS.
**Fundamentals timing**: EDGAR values are keyed to the SEC filing date and forward-filled
(preprocessing_notebooks/MMGTFFF_SECEDGAR_preprocessing.ipynb, cell 17); the profile uses
rows dated < t, so a filing on day d is first used for day d+1. When one filing reports
several periods, one value per filing date is kept (noise, not look-ahead).
**Sensitivity** (48-run validation grid): lr 1e-4 -> all 12 window/beta/width combinations
reach val MCC 0.14-0.26; lr 1e-3 -> 0.00-0.20, unstable.

## Experiment 12: disentangled MMGT (MMGT-D) -- negative result (pre-registered)

Market head on the market-summary encoder (+ BCE on market direction); stock branch with
no market input, centred over window-valid stocks, bias-free head; logit = m + e;
gradient-reversal adversary. Selected on validation (D_adv, lambda_m = 1).

| Period | MCC D / MMGT | Within-day D / MMGT | Rank IC D / MMGT | Day-collapse D / MMGT |
|---|---|---|---|---|
| Oct-Dec (10 seeds) | 0.064 / 0.146 (p = 1.4e-5) | 0.027 / 0.023 (p = 0.61) | 0.014 / 0.001 (p = 0.11) | 0.080 / 0.219 |
| A (5 seeds) | 0.074 / 0.048 | 0.043 / 0.037 | 0.033 / 0.023 | -0.002 / 0.029 |
| B (5 seeds) | 0.082 / 0.106 | 0.060 / 0.017 (day-level p = 0.002) | 0.069 / 0.045 | 0.025 / 0.174 |

Guard failed (pooled MCC significantly lower) -> negative result; paper keeps MMGT.
Market head alone: MCC 0.040. Pooled day-level bootstrap D minus MMGT: -0.083
[-0.157, -0.008]. Trading: long-short Sharpe 0.34, timing 1.68.

## Experiment 12b: two-branch MMGT-D2 -- MET its pre-registered criterion

Timing branch = a full MMGT; m_t = mean of its logits over window-valid stocks.
Selection branch = MMGT-D's market-blind centred branch (+ gradient-reversal adversary).
logit = m_t + e. Selected on validation: adversary on, lambda_aux = 1 (val MCC 0.258).

| Period | MCC D2 / MMGT | Within-day D2 / MMGT | Rank IC D2 / MMGT | Day-collapse D2 / MMGT |
|---|---|---|---|---|
| Oct-Dec (10 seeds) | **0.151** / 0.146 (p = 0.77) | **0.031** / 0.023 (p = 0.22) | **0.027** / 0.001 (p = 0.0018) | 0.205 / 0.219 |
| A (5 seeds) | 0.052 / 0.048 | 0.038 / 0.037 | 0.025 / 0.023 | 0.004 / 0.029 |
| B (5 seeds) | **0.178** / 0.106 | 0.060 / 0.017 (day-level p = 0.0005) | 0.068 / 0.045 | 0.178 / 0.174 |

Standard period: accuracy 0.572, macro-F1 0.568 +/- 0.023, AUC 0.612 (all highest).
Day-level bootstrap, Rank IC D2 minus MMGT: +0.026 [0.007, 0.047], p = 0.006; vs ALSTM +0.000.
Pooled MCC day-level vs MLP p = 0.070, LSTM 0.072, ALSTM 0.015, LR 0.025.
All 20 seed-matched runs (3 periods): Rank IC higher in 15/20 (t p = 0.005, sign p = 0.041);
within-day higher in 14/20 (t p = 0.058).
Portfolios (Sharpe): long-only 1.21, long-short 1.39 (10 bps -2.27), timing 3.38 (10 bps 2.45),
timing return +12.7%, max DD 4.3%.
Capacity control (paper/exp12b_capacity_control.json): averaging two independently trained
MMGTs (5 disjoint seed pairs, logit average) -> MCC 0.161 [0.109, 0.205], Rank IC 0.003
[-0.004, 0.024]. Parameters: MMGT 132k, MMGT-D2 273k. So D2's MCC is no better than a
two-MMGT ensemble; its Rank IC gain is not explained by capacity.

## Experiment 13: MMGT-D2 ablation and market-information probe (review round 3)

Pre-registered in PREREGISTRATION.md before any run. All 10 D2_final and 10 MMGT reruns
(with --save-states) reproduce the original test MCC exactly. Scripts:
`scripts/runs/exp13.sh`, `paper/exp13_ablation.py` -> `paper/exp13_ablation.json`,
`paper/exp13_probe.py` -> `paper/exp13_probe.json`.

**(a) Ablation (disclosed post-hoc; standard test, 10 seeds; Welch over runs vs MMGT)**

| Variant | MCC | Within-day | Day-collapse | Rank IC (p vs MMGT) |
|---|---|---|---|---|
| MMGT | 0.146 +/- 0.029 | 0.023 | 0.219 | 0.001 |
| 2xMMGT ensemble (5 pairs) | 0.161 +/- 0.040 | -- | -- | 0.003 |
| D2_final (adv, lambda_aux 1) | 0.151 +/- 0.037 | 0.031 | 0.205 | 0.027 (0.002) |
| D2adv_la0.5 (adv, 0.5) | 0.143 +/- 0.029 | 0.027 | 0.181 | 0.023 (0.008) |
| D2_la1 (no adv, 1) | 0.155 +/- 0.027 | 0.030 | 0.173 | 0.023 (0.008) |
| D2_la0.5 (no adv, 0.5) | 0.135 +/- 0.050 | 0.026 | 0.171 | 0.017 (0.036) |

The Rank IC gain comes from the two-branch centred head; the adversary adds little. No
variant changes MCC (p >= 0.49). Timing-branch logits l have Rank IC -0.004 to 0.008 in
every variant (the logit's Rank IC equals e's by construction, since m is constant per day).

**(b) Probe** (AUC on 64 test days of the seed-averaged probe score, 95% day bootstrap)

| Representation | Next-day market | Yesterday's market | 10-day trend |
|---|---|---|---|
| Market-index input | 0.588 [0.44, 0.74] | 0.993 | 1.000 |
| MMGT states | 0.692 [0.54, 0.83] | 0.990 | 0.983 |
| D2 timing branch | 0.670 [0.53, 0.81] | 0.999 | 0.978 |
| D2 selection, centred | 0.633 [0.48, 0.78] | 0.968 [0.92, 1.00] | 1.000 |
| D2_la1 selection, centred (no adv) | 0.613 [0.46, 0.75] | 0.987 | 0.996 |

Next-day direction is near chance for the market input itself (64 days: no power).
The centred selection states encode yesterday's market and the trend. Per the
pre-registered rule, MMGT-D2 is described as separating timing and selection at the
output, NOT as disentangled. Paper title changed accordingly.
