# MMGTFFF — Consolidated findings across nine experiments

**Scope.** 87 StockNet tickers, protocol-A splits, test set fixed at **3,720
samples** in every experiment so all numbers below are directly comparable.
Primary metric: MCC. Where it matters, **within-day MCC** (per-day
cross-section, thresholded at that day's median) is reported alongside, because
48.3% of test label variance is between-day market direction.

This document is the summary. Each experiment has its own write-up with full
method, controls and disclosures.

---

## 1. The three findings

**1. Next-day *direction* is not predictable from this data.**
Seven architectures — logistic regression, LSTM, MLP, hierarchical attention,
MAN-SF-style bilinear, GAT, graph transformer, RGCN, GraphSAGE, and a
macro-conditioned tweet graph. Best honest result: **+0.0338 MCC**, against a
seed-to-seed noise floor of ~0.030. Nothing survives that clears its own noise.

This holds for the graph work specifically. The best graph result (Exp 6, RGCN,
pooled +0.0598) is the project's highest neural number and it *replicates* — but
its within-day MCC is +0.0046 (p=0.43), so it is calling market direction, not
picking stocks. No graph in this project does cross-sectional work.

**2. Next-day *volatility* is predictable — but not by us.**
The identical pipeline, pointed at "will tomorrow's absolute move exceed the
training median", scores **+0.1433 MCC / 58.5% accuracy**. But **EWMA
(RiskMetrics, 1994)** — one parameter, λ=0.94 — scores **+0.1475 pooled and
+0.1966 within-day**, beating the LSTM on every metric. No neural model built in
this project beat a classical baseline on either target.

**3. Financial text contributes nothing, and degrades a good model.**
Every apparent text gain across nine experiments traced back to tweet *volume*,
press *coverage*, or feature/label *timing* — never to semantic content. Added
to the best baseline, eight dimensions of FinBERT cost **37% of its MCC**
(+0.1475 → +0.0925).

---

## 2. Four results that looked real and were not

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

- A rigorous demonstration that **semantic financial text adds no measurable
  predictive value** for next-day movement on the StockNet benchmark, across
  nine experiments and six architecture families, with the mechanism behind each
  apparent counter-example identified rather than assumed.
- A demonstration that **the target, not the model, sets the ceiling** —
  volatility is predictable where direction is not, using identical features.
- A demonstration that **classical baselines beat deep models** on the one target
  that is predictable.
- A methodological contribution: **four documented cases of results that passed
  standard significance machinery and were false**, with the specific cheap
  control that caught each.

Not claimable:

- Any improvement over published StockNet or MAN-SF numbers.
- Any benefit from graph structure, macro conditioning, or text semantics.
- MAN-SF reproduction — its Wikidata company graph was never built here.

---

## 6. Open items

1. **Obtain `stocknet-dataset/price/preprocessed/`.** The volatility result is
   leak-free and replicated, but the per-feature correlation with |next move| is
   *negative* where clustering predicts positive, and `data[3:6]` cannot be
   identified without the raw files. The result stands; the explanation does not.
2. ~~Re-run Experiment 6's +0.0677 at 5 seeds.~~ **Done — §2, §3. Pooled holds;
   within-day is zero.**
3. **Federated learning is 0% built.** It is in the project title and no code
   exists. It will not improve accuracy — federated training typically costs a
   little versus centralised — so it is a completeness deliverable, not a
   performance one.
4. **Longer horizons** (5-day, 10-day volatility) are the most promising
   remaining direction, and must be benchmarked against EWMA, never against
   price-only.
