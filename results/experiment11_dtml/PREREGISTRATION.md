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
