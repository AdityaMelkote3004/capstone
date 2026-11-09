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
