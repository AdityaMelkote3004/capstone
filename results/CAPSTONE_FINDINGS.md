# MMGTFFF — Consolidated findings across eleven experiments

**Scope.** 87 StockNet tickers, test set fixed at **3,720 samples** in every
experiment (verified row by row between the two preprocessings used), so all
numbers below are directly comparable. Primary metric: MCC. Where it matters,
**within-day MCC** (per-day cross-section, thresholded at that day's median) is
reported alongside, because a market-direction oracle alone scores pooled MCC
0.625 on this test set.

This document is the summary. Each experiment has its own write-up with full
method, controls and disclosures. **The paper (`paper/main.tex`) reports
Experiment 11; its numbers are in `results/experiment11_dtml/RESULTS.md`.**

---

## 0. The final result (Experiment 11 — the paper)

**MMGT** — the DTML data-axis transformer (every stock attends to every other
stock; a shared market context) extended with a 47-feature multi-modal stock
profile (volatility, SEC EDGAR fundamentals, tweet volume, graph-derived peer
features) — reaches **pooled MCC 0.146 ± 0.029 over ten seeds**, above every
baseline trained on the same split (baselines re-tuned on validation in the
revision; best 0.060) and above our own earlier models. Run-level p < 1e-6 in every
comparison; at the level of test days (day-block bootstrap) the advantage is
significant against ALSTM and LR, borderline against the tuned MLP (p = 0.075) and
LSTM (p = 0.059). It was the model pre-registered as "ours" before any test result
existed. Current numbers: `results/experiment11_dtml/RESULTS.md`.

The organising finding is that pooled MCC mixes two skills:

| | Pooled | Within-day (stock selection) | Day-collapse (market timing) |
|---|---|---|---|
| per-stock LSTM (tuned) | 0.055 | **0.025** | 0.082 |
| MMGT-base (DTML backbone) | 0.139 | 0.008 (n.s.) | 0.130 |
| **MMGT** | **0.146** | 0.023 | **0.219** |

MMGT's gain is mostly market timing: its timing portfolio has Sharpe 4.2 (3.2
after 10 bps) against 1.4 for buy-and-hold, while its long-short portfolio is flat
and the per-stock ALSTM is the best stock picker in money terms. The multi-modal
profile adds stock selection (within-day 0.008 → 0.031, run-level p = 0.011,
day-level p = 0.14), and MMGT's selection gain over the backbone has the same sign
in all three walk-forward periods (15/20 paired runs, sign test p = 0.041).
Pooled-MCC rankings shift between periods with market timing. Hand-built graphs
inside attention add nothing. Federated training (150 rounds) matches centralised
pooled MCC (0.131 vs 0.115, p = 0.67) but loses stock selection (p = 0.037).

This revises Finding 1 below, which was written before Experiment 11.

---

## 1. The three findings of Experiments 1–9 (revised where noted)

**1. Next-day *direction* — not predictable by Experiments 1–9's models, but
predictable at the market level (revised by Experiment 11).**
Seven architectures — logistic regression, LSTM, MLP, hierarchical attention,
MAN-SF-style bilinear, GAT, graph transformer, RGCN, GraphSAGE, and a
macro-conditioned tweet graph. Best honest result: **+0.0338 MCC**, against a
seed-to-seed noise floor of ~0.030.

The best graph result (Exp 6, RGCN, pooled +0.0598) replicates, but its
within-day MCC is +0.0046 (p=0.43): it calls market direction, not stocks.
*Experiment 11 shows that a transformer with a shared market context does this
far better (pooled 0.139) and that adding the multi-modal profile restores stock
selection (Section 0).*

**2. Next-day *volatility* is predictable — but not by us.**
The identical pipeline, pointed at "will tomorrow's absolute move exceed the
training median", scores **+0.1433 MCC / 58.5% accuracy**. But **EWMA
(RiskMetrics, 1994)** — one parameter, λ=0.94 — scores **+0.1475 pooled and
+0.1966 within-day**, beating the LSTM on every metric. No neural model built in
this project beat a classical baseline on either target.

**3. Financial text contributes nothing, and degrades a good model.**
Every apparent text gain across Experiments 1–9 traced back to tweet *volume*,
press *coverage*, or feature/label *timing* — never to semantic content. Added
to the best baseline, eight dimensions of FinBERT cost **37% of its MCC**
(+0.1475 → +0.0925).

---

## 2. Six results that looked real and were not

The value of this project is concentrated here. In each case the apparent
finding passed the checks that were being applied at the time, and failed a
check that was added afterwards.

| # | Apparent result | What killed it | Detection cost |
|---|---|---|---|
| Exp 1 | LSTM + tweet counts, **+0.0922** MCC — the project's reference point for two years of work | Single unreplicated run. Matched-architecture rerun at 5 seeds: **mean +0.0017** | 1 hour |
| Exp 7 | Macro-conditioned tweet graph, **+0.0425** within-day, survived Bonferroni, AUC agreed, 5/5 seeds | News features windowed to *t*'s 09:30 open, but the label is close-to-close, so the channel **overlapped the label in time**. Removing it: graph advantage +0.0305 → **−0.0032** | 2 hours |
| Exp 8 | PCA-compressed text, **+0.0424** over price-only, significant on all four metrics | PC1 correlated **−0.986** with "row has no tweets". The gain was **missingness encoding**, not semantics. Semantics beyond presence: +0.0088, p=0.13 | 40 min |
| Exp 9 | Volatility, **+0.1433** MCC, leak-checked, replicated | **EWMA beats it** on every metric | 30 min |
| Exp 6 | RGCN sector+correlation, **+0.0677** — the best neural number in the project | Survived a 5-seed top-up (+0.0598), then **within-day MCC came in at +0.0046, p=0.43** — below the no-graph baseline. It ranks *days*, not *stocks* | 45 min |
| Exp 10 / 11 | Graph-derived peer features: Exp 10 volatility **+0.0287** (p=0.005); Exp 11 first profile | The feature table holds only labelled rows, so a row's **presence on the target day revealed whether that stock's move left the buffer zone** (perfect 26,584/16,782 split). Peer neighbour count correlated **0.947** with the number of neighbours labelled that day. Caught in review before the paper; the strictly-prior rebuild passes a truncation test, and Exp 11's results are reported from it. Exp 10's +0.0287 is void | 1 hour |

**The pattern.** Every one of these was caught by a *cheap* control that was not
part of the original protocol. None was caught by significance testing, seed
averaging, multiple-comparison correction, or the leakage tripwire — all of
which the experiments passed.

**The transferable lesson:** on a benchmark where half the label variance is
market-wide and the effect sizes are near the noise floor, standard statistical
machinery is *not sufficient* to establish that a model predicts anything. The
controls that mattered were domain-specific: replicate across seeds, check
feature/label timing, check what your leading components actually encode, and
run the classical baseline.

---

## 3. Full result table

Pooled MCC on the fixed 3,720-sample test set.

| Experiment | Best configuration | MCC | Status |
|---|---|---|---|
| 1 — StockNet protocol A | LSTM + tweet counts | +0.0922 | ❌ does not replicate (+0.0017 at 5 seeds) |
| 2 — Hierarchical attention | VADER + logistic | +0.049 | below Exp 1's claim, itself invalid |
| 3 — MAN-SF bilinear | price + fundamentals, **no text** | +0.033 | text made it worse |
| 4 — GAT | sector graph | +0.0160 | over-smoothing diagnosed and fixed |
| 5 — Graph transformer | — | ~+0.013 | no improvement |
| 6 — Dense graph grid | RGCN sector+correlation | +0.0598 | ❌ pooled holds at 5 seeds, but within-day **+0.0046 (p=0.43)** — market-direction calling |
| 7 — Macro-conditioned tweet graph | tweet graph, no conditioning | +0.0412 | ❌ effect was label-timing overlap |
| 8 — Low-dimensional text | counts + PCA-2 | +0.0338 | ⚠️ gain is missingness encoding |
| 9 — **Volatility target** | **EWMA (classical)** | **+0.1475** | ✅ holds; neural models lose to it |
| 10 — Multi-modal graph (volatility) | peer features | +0.2261 | ❌ void: target-day leak in peer features |
| 11 — **MMGT (the paper)** | **DTML backbone + multi-modal profile + ranking term** | **+0.146** | ✅ 10 seeds, pre-registered, leak-tested; within-day +0.023 |

**Caveat on the graph experiments (4–7, 10), found during Experiment 11.** Their
daily graphs were built from the exact-protocol table, which holds only labelled
rows, so each day's node set contained only stocks whose move on the target day
left the buffer zone. Message passing over that node set sees target-day move
sizes. Most of these results were null or already invalidated for other reasons;
Exp 6's RGCN (+0.0598) is used in the paper only as a baseline, flagged there as
possibly optimistic. Experiment 11's models never condition on the target-day
node set.

Reference points: market-direction oracle **+0.6248** (knows only each day's
direction — shows how much pooled MCC rewards non-stock-specific skill);
perfect per-day ranker, within-day **+0.5152**.

**Note on Exp 6.** Now fully checked. The pooled figure survived a 5-seed top-up
(+0.0677 → +0.0598, 5/5 positive) — unlike Exp 1's, it does not collapse. But
Exp 6 only ever logged *pooled* MCC, and adding within-day settles it:

| | mean | positive seeds | vs zero |
|---|---|---|---|
| pooled | +0.0598 | 5/5 | — |
| **within-day** | **+0.0046** | 3/5 | **p = 0.43** |

+0.0046 is indistinguishable from zero and *below* the price-only no-graph
baseline (+0.0088). The model ranks **days**, not **stocks**; the graph does no
cross-sectional work. This went undetected for the whole project because pooled
MCC was the only metric recorded.

---

## 4. Why "more information" made things worse

The project's premise was that adding modalities — text, news, graph structure —
would improve prediction. It consistently did the opposite, for three
identifiable reasons.

**Averaging destroys signal.** Mean-pooling 64 tweets into one vector is like
averaging 64 faces: the distinctive features cancel. If four tweets are
informative and sixty are noise, the mean is 94% noise. Tweet *count* survives
because a count cannot be diluted by averaging — which is exactly why the
simplest feature beat the sophisticated one, repeatedly.

**Dimensions grow, data doesn't.** Price-only is ~15 inputs. With FinBERT it is
1,536 — a 100× increase against a fixed 20,344 training rows. Every added
dimension is another parameter fit to the same evidence. §2.3 of Experiment 9
measures the cost directly: 8 extra dimensions cost a working model 37% of its
performance.

**Public information is already priced.** If a tweet reliably signalled tomorrow's
move, traders would act on it today and the price would move today. The signal
deletes itself. This is not a modelling failure; it is the efficient-market
hypothesis, and every null result here is consistent with it.

---

## 5. What the capstone can claim

Defensible, with evidence:

- **MMGT has the highest pooled MCC on the standard test period**, above every
  tuned baseline and our earlier models: 0.146 ± 0.029 over ten seeds (run-level
  p < 1e-6; day-level significant vs ALSTM and LR, borderline vs MLP and LSTM).
- **Pooled MCC mixes market timing and stock selection**, and the two can be
  separated: MMGT's gain is mostly timing (timing-portfolio Sharpe 4.2 vs 1.4
  buy-and-hold); the profile adds a small stock-selection gain that is consistent
  in sign across three test periods.
- **Rankings by pooled MCC are period-dependent** (walk-forward): MMGT is best in
  Oct–Dec, below MMGT-base in Jun–Jul, level with LR in Aug–Sep.
- **Graph structure helps as peer features, not inside attention**: graph-biased
  attention adds nothing (p = 0.45), graph-masked attention is worse.
- **Federated training** across sector clients matches centralised pooled MCC
  with a 150-round schedule (0.131 vs 0.115, p = 0.67) but loses stock selection
  (within-day −0.004 vs 0.024, p = 0.037).
- **Semantic text adds nothing measurable** (Experiments 2, 3, 7, 8); tweet
  *volume* is what carries signal, which is why MMGT uses counts.
- A methodological contribution: **six documented cases of results that passed
  standard significance machinery and were false**, each with the control that
  caught it.

Not claimable:

- A new state of the art: the backbone is DTML's, and our reproduction (0.139)
  is below its published 0.191.
- Any benefit from macro conditioning or text semantics.
- MAN-SF reproduction — its Wikidata company graph was never built here.

---

## 6. Open items

1. ~~Explain the negative volatility correlation.~~ **Resolved in Experiment 10:**
   properly computed EWMA volatility correlates +0.27 with |next move|, as
   clustering predicts; the negative sign was an artefact of StockNet's own
   Price1/2/3 channels.
2. ~~Re-run Experiment 6's +0.0677 at 5 seeds.~~ **Done — §2, §3. Pooled holds;
   within-day is zero.**
3. ~~Federated learning is 0% built.~~ **Done — Experiment 11 stage F**
   (`scripts/run_dtml_federated.py`): FedAvg over nine sector clients. With the
   150-round schedule (revision) it matches centralised pooled MCC and loses
   stock selection. Open: FedProx or cross-sector aggregation to keep selection.
4. **Longer horizons** (5-day, 10-day volatility) are the most promising
   remaining direction, and must be benchmarked against EWMA, never against
   price-only.
5. **Explainability** (in the original scope) is not done: attention maps over
   stocks exist in the model but have not been validated.
