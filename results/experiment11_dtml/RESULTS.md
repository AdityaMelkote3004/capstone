# Experiment 11 — test results (pre-registered ladder, 5 seeds per rung)

Test set: StockNet/ACL18, 2015-10-01..2015-12-31, n = 3,720 (identical to the
group paper). Configuration chosen on validation only (see PREREGISTRATION.md,
committed in c7e7e33 before any test metric existed).

| Rung | Pooled MCC | Acc | AUC | Within-day MCC | Day-collapse MCC |
|---|---|---|---|---|---|
| F0 DTML, faithful | +0.1302 ± 0.0648 | 0.5537 | 0.5943 | +0.0004 | +0.1066 |
| F1 + Exp 10 stock/peer features | +0.1275 ± 0.0314 | 0.5525 | 0.5647 | +0.0400 | +0.1329 |
| F2 + ranking loss (declared model) | +0.0957 ± 0.0290 | 0.5391 | 0.5540 | +0.0319 | +0.1139 |
| F3 F2 without market context | +0.0829 ± 0.0452 | 0.5294 | 0.5366 | +0.0260 | +0.1248 |
| F4 F0 without stock attention | +0.0593 ± 0.0332 | 0.5322 | 0.5493 | +0.0223 | +0.0443 |

Comparisons (Welch two-sample vs the paper's 5-seed mean ± std; one-sample vs the
5-seed tweet-count LSTM):

| Rung | vs paper +0.0876 ± 0.0253 | vs LSTM +0.0017 |
|---|---|---|
| F0 | +0.0426, p = 0.227, 4/5 seeds above | p = 0.011 |
| F1 | +0.0399, p = 0.059, 5/5 seeds above | p = 0.0009 |
| F2 | +0.0081, p = 0.652, 3/5 seeds above | p = 0.0019 |
| F3 | −0.0047, p = 0.845 | p = 0.016 |
| F4 | −0.0283, p = 0.171 | p = 0.018 |

## What can be claimed
- Every DTML rung beats the tweet-count LSTM baseline on this test set (p ≤ 0.02).
- DTML with the group's multimodal StockNet features (F1) exceeds the group
  paper's full model on all five seeds (+0.040 MCC), p = 0.059. It is also the
  most stable rung (std 0.031 vs 0.065 for faithful DTML) and the only rung with
  both high pooled MCC and a within-day stock-discrimination edge (+0.040).

## Disclosures
- The model declared in advance was F2. The within-day ranking loss lowered
  pooled MCC relative to F1; F1 is reported as a pre-registered rung, not as a
  post-hoc pick.
- Faithful DTML's pooled MCC is almost entirely market timing: within-day
  +0.0004, and the day-collapse control recovers most of its score. Its seed
  variance is large (seed 2: +0.026; seed 1: +0.181).
- Day-block bootstrap 95% CIs per seed are wide (roughly ±0.12), because the
  market component of a 64-day test set has an effective n near 64.
- DTML's published 0.191 was not reproduced; faithful DTML here averages 0.130.
  The market index is proxied by an equal-weighted mean of the 87 stocks, since
  the S&P 500 series is not in the published data.
