# Experiment 11 — final test ladder (pre-registered before test was touched)

Written 2026-11-09, after grid A (48 dev-only runs) and stage C (12 dev-only runs).
No test metric had been computed for any configuration at this point.

Config, chosen on validation MCC by DTML's own selection rule (paper Sec 4.1):
window 10, beta 1.0, hidden 64, lr 1e-4, heads 4, dropout 0.15, selective reg 1.0.
Configurations tried on dev before this file: 48 (grid A) + 6 rungs x 2 seeds (stage C).

Rungs, 5 seeds each (0-4), checkpoint selected on val MCC, all reported:
  F0  DTML, faithful reproduction (price features only)
  F1  F0 + Exp 10 stock features with peer/sector block (all StockNet modalities)
  F2  F1 + within-day pairwise ranking loss (alpha = 1)     <-- declared "our model"
  F3  F2 without the market context (ablation)
  F4  F0 without stock attention (DTML's own ablation)

Primary claim: F2 pooled test MCC (5-seed mean +/- std) versus
  - tweet-count LSTM, 5 seeds, this test set:  +0.0017
  - group paper's full model, 5 seeds:         +0.0876
Also reported for every rung: accuracy, AUC, within-day MCC, day-collapse MCC,
day-block bootstrap 95% CI.

## Addendum (written after the 5-seed test results, before running more)
Rungs F0, F1 and F2 are extended to 10 seeds (5-9 added), matching DTML's own
10-seed protocol. No configuration changes. This decision was made after seeing
the 5-seed results (F1 vs paper p = 0.059), so it is disclosed as a post-hoc
extension; all three rungs are extended equally and all 10 seeds are reported.

## Stage D — reduced feature sets (declared before running)
Feature indices: 0 z_open, 1 z_high, 2 z_low, 3 z_close, 4 z_adj_close,
5-10 z_d5, z_d10, z_d15, z_d20, z_d25, z_d30.
  S0 all 11            (reference, same seeds)
  S1 returns only      3
  S2 returns+intraday  0 1 2 3
  S3 trend only        5-10
  S4 return+trend      3 5-10
  S5 compact           3 5 7 10
Screen on validation, seeds 0-2, config as above. The subset with the best
mean val MCC among S1-S5 goes to test at 10 seeds; S0 is the reference. If no
reduced subset is within the S0 val range, the reduced model is reported as not
viable rather than tuned further.

## Stage E — graph-aware DTML (declared before running)
All variants = F1 (DTML + Exp 10 stock/peer features, same config) plus a graph
in the stock-attention layer:
  G1 bias, relations sector + correlation + comention
  G2 bias, relations sector + correlation + comention + residual (group paper's
     market-residual top-8 graph, built from ACL18 z_close, 20 prior days)
  G3 mask (attend only to neighbours + self), same four relations
Screen on validation, seeds 0-2. The variant with the best mean val MCC goes to
test at 10 seeds ONLY IF that mean is at least F1's validation mean on seeds
0-2; otherwise the result is reported as "the explicit graph did not help" and
nothing goes to test. Test comparison: paired by seed against F1.

## Stage F — federated training (declared before running)
F1's model, features and config trained with FedAvg over the nine sector
clients (7-10 tickers each), 1 local epoch per round, up to 40 rounds, global
model selected on validation MCC (patience 10), 5 seeds. Compared, paired by
seed, against centralised F1. Expected direction: at or below F1 (federation
adds no data). A one-round smoke run was executed to check the code and
discarded; no selection was made from it.

## Correction (2026-11-10): target-day leak in the stock profile
After the test results above, a review found that the v1 profile keyed every
feature to the TARGET date in the modeling parquet, which only contains
labelled rows. A row exists at (stock, t) exactly when that stock's move on t
left StockNet's buffer zone (26,584 / 16,782 cells, zero exceptions), so row
presence on t is target-day information. It reached the model through the peer
neighbour count (corr 0.947 with the number of neighbours labelled at t), the
peer means, and zero-filled profiles of unlabelled stocks.

All results that use the profile -- F1, F2, F3, G2 and the federated runs -- are
therefore void. Results that do not use it (F0, F4, S4, all baselines) are
unaffected.

Fix: profile v2 (src/data/acl18_extra.py) uses only data dated strictly before
t. tests/test_profile_leakage.py rebuilds it with all data from t onward deleted
and finds zero difference on 12 sampled days across train/val/test.

Rerun with v2, identical configurations and seeds, tags suffixed "c":
F1c, F2c (10 seeds), F3c (5 seeds), G2c (10 seeds), FEDc (5 seeds). No
hyperparameter or selection changes. The paper reports only the v2 results.

## Post-hoc diagnostic (2026-11-10): one-epoch centralised control
After the federated results, a review noted that 4 of 5 federated seeds kept
their round-1 global model. To test whether the federated pattern reflects
partitioning or under-training, the centralised profile model (F1c config) was
trained for exactly one epoch (tag UT1c, 5 seeds) and two epochs (UT2c; identical
selection, so identical results). Pooled 0.053, within-day 0.016, day-collapse
0.035 -- the same pattern as federated (0.040 / 0.027 / -0.024). This diagnostic
informed only the paper's interpretation; no model was selected from it.
The leakage test was also extended to rebuild the correlation and co-mention
edges from truncated data; they match the cache on all 12 sampled days.

## Revision after external review (2026-11-11) — declared before running
1. Baselines tuned on validation: LR C in {0.01,0.1,1,10}; MLP hidden {32,64,128}
   x lr {1e-3,1e-4}; LSTM and ALSTM hidden {32,64,128} x lr {1e-3,1e-4} x
   window {10,15}; 2 seeds each; best mean val MCC is the tuned config.
2. Final test runs with every prediction saved (all window-valid stocks, labelled
   or not): tuned baselines, MMGT-base (F0 config), MMGT (F2c config) and the
   profile model (F1c config), 10 seeds, OMP_NUM_THREADS=2. Paper numbers come from
   these runs.
3. Date-level statistics: paired day-block bootstrap (resampling test days) of the
   seed-averaged MCC difference, MMGT vs each model; 95% CI.
4. Long-short backtest on saved predictions: each test day, equal-weight long the
   top-10 and short the bottom-10 stocks by predicted probability among all
   window-valid stocks; realised close-to-close returns; cost 0 and 10 bps per side.
   Reported for every model, whatever it shows.
5. Walk-forward: two extra folds with the same configurations, no re-tuning,
   5 seeds: fold A train to 2015-03-31, val Apr-May 2015, test Jun-Jul 2015;
   fold B train to 2015-05-29, val Jun-Jul 2015, test Aug-Sep 2015.
6. Federated, longer schedule: FedAvg, 9 sector clients, up to 150 rounds,
   patience 30, val MCC logged every round, 5 seeds.
   DECISION RULE: federated stays in the paper if its mean test MCC >= 0.10 AND
   it is not significantly below the centralised profile model on the same seeds
   (paired p >= 0.05). Otherwise it is removed and moved to future work.

### Revision -- baseline tuning result (validation only, recorded before any test run)

Selected by mean validation MCC over seeds 0-1 (LR is deterministic, 1 run):

| Baseline | Grid | Selected | Val MCC |
|---|---|---|---|
| Logistic regression | C in {0.01, 0.1, 1, 10} | C = 0.01 | 0.119 |
| MLP | hidden {32,64,128} x lr {1e-3,1e-4} | hidden 64, lr 1e-3 | 0.166 |
| LSTM | window {10,15} x hidden {32,64,128} x lr {1e-3,1e-4} | window 10, hidden 64, lr 1e-3 (tie with w15 broken toward the shorter window) | 0.075 |
| Attention LSTM | same grid | window 15, hidden 128, lr 1e-4 | 0.088 |

Final test runs (10 seeds, 0-9, predictions saved): R_lr, R_mlp, R_lstm, R_alstm with the
settings above; R_base (price-only, window 10, beta 1, hidden 64, lr 1e-4); R_profile
(+ peer profile); R_mmgt (+ peer profile + ranking loss, alpha 1).
Walk-forward folds A and B: R_mmgt, R_base, R_mlp, R_lstm, seeds 0-4.

Federated note: FEDlong seed 0 finished first (best round 0, test MCC 0.025). The decision
rule above is applied unchanged once all 5 seeds finish.

### Revision -- federated decision (rule applied unchanged)

FEDlong, seeds 0-4: test MCC 0.025, 0.164, 0.130, 0.157, 0.177 (mean 0.131).
Centralised profile model on the same seeds: mean 0.115; paired p = 0.67.
Rule: mean >= 0.10 AND not significantly below centralised -> **met; federated stays.**
Reported alongside: within-day MCC drops 0.024 -> -0.004 (paired p = 0.037).

## Experiment 12 -- disentangled MMGT (MMGT-D), pre-registered before any run

Motivation: external review; MMGT's gain is mostly market timing. MMGT-D is a
follow-up designed AFTER MMGT's test results were known, so it is reported as a
separate model, never as a renamed MMGT. Prior work: ADD (Tang et al., 2020,
adversarial market/excess disentanglement), two-step disentanglement (Hadad et al.,
CVPR 2018), FactorVAE (Duan et al., AAAI 2022).

Model (src/models/dtml_disentangled.py), same backbone/inputs as MMGT (window 10,
hidden 64, lr 1e-4, peer profile, ranking alpha 1):
- market branch: market-index encoder only -> day logit m_t; trained with an extra
  BCE on market_up_t = [mean target-day return over WINDOW-VALID stocks > 0]
  (training label only; never an input).
- stock branch: stock encoder + profile + stock attention, NO market input; output
  centred over window-valid stocks after the last layer norm, then a bias-free head
  -> e_{i,t}, which sums to exactly zero over each day.
- D-struct: logit = m_t + e_{i,t}.
- D-beta:   logit = (1 + b_i) m_t + e_{i,t}, b_i = tanh(w . h_i) from the stock branch
  (systematic response to the market).
- D-adv:    D-struct + gradient-reversal adversary predicting market_up_t from
  permutation-invariant statistics (mean |.|, std) of the centred stock states (ADD-style).
- Selective L2 on all output heads.

Validation grid (seeds 0, 1): {D-struct, D-beta, D-adv} x lambda_market {0.5, 1}
= 6 configs. Adversary weight fixed at 0.1 (not tuned).
Selection: mean validation pooled MCC (same rule as everywhere).
Test: chosen config, 10 seeds (0-9), predictions + components (m, e, b) saved.
Walk-forward: folds A and B, seeds 0-4.

Primary endpoint (stock selection), test, 10 seeds vs MMGT (R_mmgt):
  within-day MCC higher AND mean daily Rank IC (Spearman of prob vs target-day
  return over window-valid stocks) higher, with run-level Welch p < 0.05 on at
  least one of the two and the other not lower.
Guard: pooled MCC not significantly below MMGT (Welch p >= 0.05 or higher mean).
Outcome: endpoint + guard met -> MMGT-D is presented as the paper's disentangled
variant; otherwise it is reported as a negative result and the paper keeps MMGT.
Day-level bootstrap and walk-forward are reported either way.

### Experiment 12 -- selection (validation only, recorded before any test run)

Grid means over seeds 0-1 (val MCC / val within-day): D_adv_lm1 0.146 / 0.084;
D_struct_lm1 0.139 / 0.039; D_beta_lm1 0.137 / 0.069; D_beta_lm0.5 0.133 / 0.074;
D_adv_lm0.5 0.129 / 0.072; D_struct_lm0.5 0.125 / 0.070.
Selected by the rule (highest mean val MCC): **D-adv, lambda_market = 1** (tag D_final).
Test: seeds 0-9 standard split; folds A, B seeds 0-4.
Also: exhaustive leakage test (all 504 target days) PASSED.

### Experiment 12 -- outcome (rule applied unchanged)

D_final vs MMGT, test, 10 seeds: pooled MCC 0.064 vs 0.146 (Welch p = 1.4e-5) -> GUARD FAILED.
Within-day 0.027 vs 0.023 (p = 0.61); Rank IC 0.0135 vs 0.0007 (p = 0.11) -> endpoint not met.
**MMGT-D is a negative result; the paper keeps MMGT.** Selection gains have the same sign in
all three periods (within-day +0.004/+0.007/+0.043, Rank IC +0.013/+0.010/+0.024; fold B
within-day day-level p = 0.002). Market head alone: MCC 0.040 -- the index encoder cannot
reproduce MMGT's timing, which therefore comes from attention pooling all stocks' histories.
Clarification (added after review of the outcome entry): the mechanism sentence above is
too strong. The market summary is itself the equal-weight index of all stocks. What is
established: MMGT's timing needs the market summary AND stock attention together (removing
either lowers MCC: 0.131 -> 0.078, 0.130 -> 0.059); a separate linear head on the index
encoding alone reaches 0.04. Validation already showed the timing loss (D_final val MCC
0.146 vs MMGT 0.211); the grid compared D variants only with each other.
No follow-up variant was run.

## Experiment 12b -- two-branch MMGT-D2 (second and LAST follow-up), pre-registered before any run

Designed after Experiment 12's test results (standard period AND folds A, B were seen),
at the user's request to keep MMGT's market timing while improving stock selection.
Diagnosis behind it: MMGT-D's index-only market head lost the timing (MCC 0.040).
A shared-backbone split was rejected: MMGT's linear output already equals
mean-part + centred-part with tied weights, so it would re-test MMGT.

Model (DTMLTwoBranch): timing branch = a complete MMGT (market summary, stock attention,
peer profile, beta 1); m_t = mean over WINDOW-VALID stocks of its logits. Selection
branch = MMGT-D's stock branch (no market input, centred, bias-free head, optional
gradient-reversal adversary) -> e_{i,t}, sum zero per day. logit = m_t + e_{i,t}.
Loss: BCE(logit) + lambda_aux * [BCE + RankNet on the timing branch's own logits]
+ RankNet(e) + 0.1 * adversary BCE (if on) + selective L2 on all heads.

Validation grid (seeds 0, 1): lambda_aux {0.5, 1} x adversary {on, off} = 4 configs.
Selection: mean validation pooled MCC.
Test: seeds 0-9 standard; folds A, B seeds 0-4.
Primary endpoint and guard: IDENTICAL to Experiment 12 (vs MMGT, 10 seeds:
within-day MCC and Rank IC both not lower, one with Welch p < 0.05; pooled MCC not
significantly lower).
Secondary (pre-registered here, chosen after Exp 12): paired test over all 20 runs
(standard + A + B, matched by seed) of within-day MCC and of Rank IC vs MMGT.
STOP RULE: whatever the outcome, this is the last disentanglement attempt; both
follow-ups are reported in the paper.

### Experiment 12b -- selection (validation only, recorded before any test run)

Grid means over seeds 0-1 (val MCC / within-day / collapse): D2adv_la1 0.258 / 0.027 / 0.205;
D2adv_la0.5 0.257 / 0.043 / 0.221; D2_la1 0.249 / 0.044 / 0.216; D2_la0.5 0.245 / 0.056 / 0.242.
Selected (highest mean val MCC): **two-branch + adversary, lambda_aux = 1** (tag D2_final).
(For reference, MMGT's val MCC: 0.211.)

### Experiment 12b -- outcome (rule applied unchanged)

D2_final vs MMGT, test, 10 seeds: pooled MCC 0.151 vs 0.146 (p = 0.77) -> guard MET.
Within-day 0.031 vs 0.023 (p = 0.22, not lower); Rank IC 0.0271 vs 0.0007 (Welch p = 0.0018)
-> **primary endpoint MET.** Day-level bootstrap of the Rank IC gain: +0.026
[0.007, 0.047], p = 0.006. Secondary (20 seed-matched runs, all periods): Rank IC higher
in 15/20 (t p = 0.005, sign p = 0.041); within-day higher in 14/20 (t p = 0.058).
Fold A: MCC 0.052 vs 0.048, IC 0.025 vs 0.023. Fold B: MCC 0.178 vs 0.106, IC 0.068 vs 0.045.
Per the rule, MMGT-D2 is presented as the paper's disentangled model, with the disclosure
that it is the second follow-up designed after test results were seen.

## Experiment 13 -- MMGT-D2 ablation and market-information probe (review round 3), pre-registered before any run

Requested by an external review after the paper's MMGT-D2 results were known. Nothing here
changes the headline model: **D2_final (two-branch + adversary, lambda_aux = 1) stays the
paper's model whatever the numbers below show**, because it was selected on validation.

(a) Ablation (DISCLOSED POST-HOC). The three MMGT-D2 configurations NOT selected in
Experiment 12b (D2_la0.5, D2_la1 without adversary; D2adv_la0.5) are run on the standard
test period, seeds 0-9, exactly as D2_final. Reported side by side with MMGT, the
two-MMGT ensemble control (Exp 12b) and D2_final; MCC, within-day MCC, day-collapse MCC,
Rank IC. No winner is picked from this table.

(b) Probe. D2_final (seeds 0-9) and MMGT (R_mmgt, seeds 0-9) are retrained with the
exact original settings and thread counts, with --save-states (states saved after
training, in eval mode, no gradients; no effect on training). The ablation runs also save
states. Reproduction is checked by comparing each rerun's test MCC with the original CSV;
if they differ, the probe is reported as a property of the training recipe, with the
reruns' own scores, never mixed into Table I.
Representations: MMGT final stock states; D2 timing-branch states; D2 selection-branch
states, uncentred and centred (what the excess head sees). Day feature: [mean, std] over
window-valid stocks (std only for centred). Probe: L2 logistic regression fitted on
training days, C by 5-fold blocked CV on training days, scored by AUC on the 64 test
days. Targets: next-day market direction (training label of the adversary), yesterday's
market direction and the 10-day market trend (both determined by the inputs).
Interval: 95% day-level bootstrap (2,000) of the AUC of the seed-averaged probe score.
Rule: a representation is "near chance" for a target if this interval includes 0.5.
Reference: the same probe on the raw market-index input window.
Wording rule for the paper: if D2_final's centred selection states are near chance on
all three targets, the paper may call MMGT-D2 "disentangled"; otherwise it describes it
as separating timing and selection AT THE OUTPUT and reports which targets leak.
Script: paper/exp13_probe.py -> paper/exp13_probe.json; runs: scripts/runs/exp13.sh.

### Experiment 13 -- outcome (rules applied unchanged)

Reproduction: all 10 D2_final and 10 MMGT reruns reproduce the original test MCC exactly
(max |diff| = 0.0). The probe therefore describes the paper's models.

(b) Probe, D2_final centred selection states, AUC of the seed-averaged score [95% CI]:
next 0.633 [0.48, 0.78] (near chance), yday 0.968 [0.92, 1.00] (NOT near chance),
trend 1.000 [1.00, 1.00] (NOT near chance). -> **Wording rule: MMGT-D2 is NOT called
"disentangled".** The paper describes it as separating timing and selection at the
output (e sums to zero each day by construction) and reports that the selection states
still encode past market moves.
Observation (not a change to the rule): with 64 test days the next-day target has little
power -- the raw market input itself is near chance (0.588 [0.44, 0.74]) and every centred
selection state except D2_la0.5 passes, adversary or not. Evidence rests on yday/trend.
Timing branch l: Rank IC ~ 0 in every variant (-0.004 to 0.008), as for MMGT (0.0007).

(a) Ablation, test, 10 seeds (Welch over runs vs MMGT): all four D2 variants raise Rank IC
(0.017-0.027 vs 0.0007, p 0.002-0.036); none changes MCC (0.135-0.155 vs 0.146,
p 0.49-0.80). D2_la1 (no adversary) has MCC 0.155, IC 0.023. The selection gain comes from
the two-branch, centred head; the adversary adds little on these metrics. Disclosed
as post-hoc; D2_final stays the headline model. Output: paper/exp13_ablation.json.

### Review round 4 note (no new runs)

Walk-forward period B is the standard split's validation period, on which the frozen
settings were chosen, so it is in-sample for them. From this round the paper reports B
separately and excludes it from all averages and tests. The pre-registered Exp 12b
secondary test (MMGT-D2 vs MMGT, seed-matched over all periods incl. B: Rank IC higher
in 15/20, p = 0.005) is still disclosed, but the paper now relies on the out-of-sample
version (period A + standard: 11/15, paired t p = 0.011, sign p = 0.12). The primary
endpoint (standard period) is unchanged. Script: paper/rev4_oos.py.

## Review round 5 -- statistical robustness re-analysis (no new training), rules fixed BEFORE any computation

Requested by a reviewer: the i.i.d. day bootstrap ignores temporal dependence. All
analyses below use only the saved test predictions; no model is retrained.

Primary day-level method (replaces the i.i.d. day bootstrap everywhere in the paper):
hierarchical bootstrap, 2,000 replicates; each replicate resamples (i) training runs
with replacement within each model and (ii) the 64 test days as a circular
moving-block bootstrap with block length 5 trading days. Statistic: difference of
run-averaged scores. 95% percentile interval; two-sided p = 2*min(P(d<=0), P(d>=0)),
floored at 1/2000. Sensitivity (reported, not used for claims): block length 1
(i.i.d. days) and 10, and days-only (no run resampling) with block length 5.
Applies to pooled MCC, within-day MCC, day-collapse MCC and Rank IC.
Multiplicity: MMGT vs the four tuned baselines is one family, Holm-adjusted. The
pre-registered primary endpoint for MMGT-D2 (Rank IC and within-day MCC vs MMGT) is
reported unadjusted as the confirmatory test.
"Retains pooled performance": no non-inferiority margin was pre-specified, so none is
claimed; the paper reports the interval of MMGT-D2 minus MMGT pooled MCC and the
smallest margin that interval excludes.
Additional descriptive analyses: daily market-direction evaluation (one observation
per day; score = mean predicted probability over window-valid stocks; target = mean
return over window-valid stocks > 0; AUC primary, MCC at 0.5 secondary); counterfactual
pooled MCC with stock-level differences removed (day-mean probability) vs day-level
differences removed (stock above its day's mean probability); within-day MCC
characterisation (days scored, per-day distribution, tercile and all-stock variants);
model-cluster bootstrap interval for the within-day MCC / Rank IC correlation.
Dropped before running: label-threshold sensitivity (the saved close-to-close returns do
not reproduce the adjusted-close labels exactly: 130 up-labels at <= +0.55%, 191
unlabelled outside the band), so relabelling would not be valid.
Script: paper/rev5_stats.py -> paper/rev5_stats.json.

### Review round 5 -- outcome (rules applied unchanged; paper/rev5_stats.json)

Run-level values reproduce the reported ones exactly. Under the primary method
(hierarchical, 5-day blocks): MMGT vs LR p=0.040, LSTM 0.030, ALSTM 0.006, MLP 0.098;
Holm-adjusted 0.090 / 0.090 / 0.024 / 0.098 -> only ALSTM survives. MMGT-D2 primary
endpoint (Rank IC vs MMGT): +0.026 [0.004, 0.051], p=0.027 (sensitivity: 1-day blocks
p=0.071, 10-day 0.031, days-only 0.0005). MMGT-D2 - MMGT pooled MCC +0.005
[-0.043, 0.060]: no margin pre-specified, so no non-inferiority claim; reported as the
smallest excluded loss (0.043). The paper's former i.i.d. day-level numbers are replaced
by these. The Exp 13 probe intervals keep their own pre-registered i.i.d. method.
Day-removed counterfactual (stock above its day's mean) is reported in the supplement
but not interpreted: mean-thresholding does not force a balanced daily split.

## Review round 6 -- logit/probability alignment, ranking quality, probe intervals (no new training), rules fixed BEFORE any computation

Requested by a reviewer. Inputs: saved test predictions (preds/, which hold every
window-valid stock and, for MMGT-D2, m_t and e_{i,t}) and saved hidden states. No model
is retrained. Populations: S_t (labelled) is a subset of V_t (window-valid). The metrics
as defined in the paper are NOT changed; everything below is reported alongside them.

1. Logit vs probability (MMGT-D2, standard period, 10 runs). Identity check:
   |mean_{V_t} e| per day (expected ~0 up to float error). Per run-day: mean_{S_t} e;
   |mean_{S_t} p - sigma(m_t)| and |mean_{V_t} p - sigma(m_t)| (median, 95th percentile,
   max); days per run on which the 0.5 call of sigma(m_t) differs from that of mean_{S_t} p
   and of mean_{V_t} p. Market branch scored directly: day-collapse MCC with every
   labelled stock of day t predicted 1[m_t >= 0], and daily AUC with score m_t.
   Sensitivity for all principal models (descriptive): day-collapse MCC under four
   aggregations -- mean probability over S_t (the defined metric), mean probability over
   V_t, mean logit over S_t >= 0, mean logit over V_t >= 0 (logits from the saved
   probabilities, clipped to [1e-6, 1-1e-6]; for MMGT-D2 the last equals m_t >= 0) --
   and daily AUC with the mean logit over V_t as score. Daily market target unchanged:
   M_t = 1[mean_{V_t} r_{i,t} > 0]. Intervals: round-5 primary method.
2. Ranking quality on S_t (descriptive): pairwise accuracy = per-day AUC of p over S_t
   (share of up/down pairs ordered correctly, ties count 1/2), averaged over the T' days
   with both labels; normalised within-day MCC = mean within-day MCC / mean MCC_max, where
   MCC_max is the within-day MCC of a perfect ranking under the same median rule on the
   same days (the mean of per-day ratios is not used: it explodes on lopsided days);
   per-day distribution (quartiles, share positive) of run-averaged within-day MCC.
   Intervals and the MMGT-D2 - MMGT difference in pairwise accuracy: round-5 primary
   method, exploratory.
3. Probe intervals: the Exp 13 seed-averaged probe AUCs, same fits, with a circular
   moving-block bootstrap of the test days (block 5; 10 as sensitivity; 2,000). The
   pre-registered i.i.d. intervals remain on record. Rule: if, with 5-day blocks, the
   interval of the centred selection states for yesterday's direction or the 10-day trend
   includes 0.5, the paper says so and weakens the probe conclusion.
4. Paired Rank IC, MMGT-D2 - MMGT, by seed (standard period 10 pairs; period A 5): the
   individual differences, mean, median, Wilcoxon signed-rank p, and a 95% bootstrap
   interval over seed pairs (10,000). Run-level only, exploratory; the primary endpoint
   and its test are unchanged.
5. Block length check (reported only; the primary block length of 5 is not changed):
   Politis-White automatic block length (arch, circular) for the per-day series of the
   run-averaged MMGT-D2 - MMGT Rank IC difference, MMGT's per-day accuracy, and the daily
   mean return of V_t.
6. Walk-forward averages: Table IV's out-of-sample average is the equally weighted mean
   of periods A (44 days) and standard (64 days); the day-weighted mean is reported in
   the supplement.
7. Fig. 2 intervals: round-5 primary level intervals of within-day and day-collapse MCC,
   adding the profile model and the federated model.

Confirmatory set (no new tests): the Exp 11 test ladder for MMGT, the federated
keep/remove rule, the pre-registered criteria of MMGT-D and MMGT-D2 (MMGT-D2's primary
endpoint), and the round-5 Holm family. Every other test in the paper is exploratory
and unadjusted.
Script: paper/rev6_stats.py -> paper/rev6_stats.json.

### Review round 6 -- outcome (rules applied unchanged; paper/rev6_stats.json)

1. |mean_V e| = 0 to float precision (identity holds). mean_S e: median |.| 0.039, max
   0.93. |mean_S p - sigma(m)|: median 0.016, max 0.159; over V: median 0.011, max 0.052.
   0.5 call differs from sigma(m_t) on 1-5 of 64 days per run (S), 0-1 (V). Market branch
   scored directly: day-collapse 0.179 [0.045, 0.297] vs the defined 0.205 [0.071, 0.320]
   (defined - branch +0.026 [-0.001, 0.067], p=0.07); daily AUC 0.652 vs 0.654. The paper
   now reports 0.179 for MMGT-D2's market branch. Other models without a market logit:
   aggregation changes day-collapse MCC by <= 0.015 (MMGT 0.219 -> 0.214); MMGT-D 0.080
   -> 0.040.
2. Pairwise accuracy: MMGT-D2 0.525 [0.493, 0.557], MMGT 0.521, ALSTM 0.514, LSTM 0.513,
   profile 0.529, MMGT-base 0.503, MLP 0.501, LR 0.491; D2 - MMGT +0.004 p=0.67. MCC_max
   mean 0.515 (min 0.11); normalised within-day MCC D2 0.060, MMGT 0.044. Per-day medians
   -0.014 to 0.020 for every model.
3. Probe with 5-day blocks: centred selection states yday/trend lower bounds >= 0.937;
   rule NOT triggered, conclusion unchanged. Lowest bound over all representations 0.903
   (5-day) / 0.885 (10-day). Next-day still inconclusive.
4. Paired Rank IC by seed: standard 8/10 positive, mean +0.026 [0.013, 0.038], Wilcoxon
   p=0.010; A 3/5, mean +0.001 [-0.016, 0.018]; both 11/15, p=0.015.
5. Politis-White circular block length: D2 - MMGT IC difference 5.6; MMGT accuracy 0.8;
   daily market return 1.2. Primary block length unchanged.
6. Day-weighted walk-forward averages: MCC ordering unchanged; Rank IC only near-tied
   LSTM / MMGT-base / MMGT swap.
Also this round: the trading simulation was moved from the paper to the supplement to
meet the page limit (no change to its numbers).

## Experiment 14 / review round 7 -- architectural controls (new training) and evaluation re-analysis, rules fixed BEFORE any run or computation

Requested by the round-7 review; the authors approved new training for the three
controls below (2026-11-16). **D2_final stays the paper's model whatever these show**;
the controls are disclosed post-hoc comparisons, never candidates for selection.

(a) Controls. Standard split, seeds 0-9, selected on validation MCC exactly as before,
test evaluated once, predictions saved (D2 controls also save hidden states). Settings
identical to D2_final (two-branch, adversary, lambda_aux 1, lambda_adv 0.1, rank weight 1,
reg 1, lr 1e-4, hidden 64, dropout 0.15, window 10, <= 200 epochs, patience 30) except:
  C1 D2_unc     no zero-sum constraint: the excess head reads the UNcentred selection
                states and has a bias (L2-penalised like the other head biases). The
                adversary still sees the centred states, so only the output constraint
                changes. e no longer sums to zero; logit = m_t + e.
  C2 D2_noatt   selection branch without stock-to-stock attention (each stock's state
                comes from its own price history and profile); centring, head and
                adversary unchanged. Its hidden states are probed with the Exp 13 probe.
  C3 MMGT_wide  single-branch MMGT (R_mmgt settings) with hidden size 92: 262k
                parameters, against 273k for MMGT-D2 and 132k for MMGT.
  Reproduction check: D2_final seed 0 rerun with the modified code (2 threads, as in
  Exp 13) must reproduce the original test MCC; otherwise this is reported.
Metrics: pooled MCC, within-day MCC, pairwise accuracy, Rank IC, day-collapse MCC,
market-logit day-collapse MCC, daily AUC. Tests: each control against D2_final and
against MMGT, hierarchical block bootstrap (round-5 primary method) and Welch over runs;
all exploratory (the confirmatory set is unchanged).
Wording rules (Rank IC, block bootstrap, two-sided 0.05):
  R1 constraint: "the zero-sum constraint contributes" only if D2_final - C1 > 0
     significantly; "the gain does not require the constraint" if C1 - MMGT > 0
     significantly and the D2_final - C1 interval includes 0; otherwise "not resolved".
  R2 attention: the same rule with C2 ("cross-stock attention contributes" / "does not
     require cross-stock attention" / "not resolved"). Probe: if C2's centred selection
     states are near chance (5-day block interval includes 0.5) for yesterday's
     direction or the 10-day trend, the paper says that attention is the route by which
     these states acquire market information; otherwise it says that removing attention
     does not remove it.
  R3 capacity: "capacity alone does not produce the gain" if D2_final - C3 > 0
     significantly; otherwise "not resolved by this control".

(b) Re-analysis of saved predictions (no training).
1. Market columns of Table I for every model with saved predictions, with 95% intervals
   (round-5 primary method): day-collapse MCC as defined (mean probability over S_t);
   market-logit day-collapse MCC (the mean logit over V_t >= 0, which is m_t exactly for
   MMGT-D and MMGT-D2); daily AUC of the mean logit over V_t against M_t. Paired MMGT-D2 -
   MMGT differences for the last two.
2. Pairwise accuracy, as implemented in round 6: per-day AUC of p over S_t (ties 1/2),
   macro-averaged over the T' days with both labels; neutral stocks excluded; days
   without both labels skipped. Intervals for every model.
3. Rank IC by subset: V_t (as reported), S_t (labelled stocks only) and N_t = V_t minus
   S_t (neutral band); a day enters a subset average only if the subset has >= 5 stocks.
   Intervals for MMGT, MMGT-D2 and the controls; paired MMGT-D2 - MMGT difference per
   subset. Check: every unlabelled window-valid stock has a return inside StockNet's
   band (counts reported).
4. Returns: M_t and Rank IC use the close-to-close change of the (split-adjusted,
   dividend-unadjusted) close. Sensitivity: both recomputed with the adjusted-close
   change (feature z_adj_close); differences reported.
5. Market representation (descriptive): the market-only logistic-regression timing
   reference (fitted on training days with the Exp 13 probe procedure, daily AUC on
   test days, 5-day block interval) on (i) the 10-day window of the 11-feature proxy
   (as before), (ii) the 10-day window of the equal-weighted mean return, (iii) the
   10-day window of an equal-weighted index built by cumulating the mean return: its
   daily change and its level relative to its 5- to 30-day moving averages.
6. Probe, incremental next-day information (descriptive): next-day direction probe on
   the market-proxy summary alone vs the proxy summary plus the centred selection
   states; difference of daily AUC with a 5-day block interval.
Script: paper/rev7_stats.py -> paper/rev7_stats.json; runs: scripts/runs/exp14.sh.
