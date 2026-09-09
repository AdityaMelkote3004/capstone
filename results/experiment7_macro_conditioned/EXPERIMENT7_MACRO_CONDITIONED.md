# Experiment 7 — Macro-Conditioned Micro Signal over a Tweet Graph

**Status:** Complete. 40/40 ladder cells trained (8 rows × 5 seeds).
**Run:** local CPU (PyTorch 2.11.0+cpu, no GPU), `scripts/run_experiment7_ladder.py`.
**Raw results:** `results.csv` (40 rows) + one `metrics.json` per run.
**Spec:** `docs/superpowers/specs/2026-09-08-event-conditioned-tweet-graph-design.md`

---

## 1. Objective

Experiment 6 measured every hand-designed graph (sector cliques, rolling price
correlation, tweet co-mention) and found no distinguishable improvement.
Experiment 7 changed three things:

1. The graph is **learned from tweet semantics** rather than hand-designed.
2. Node features carry **text** — Experiments 4, 5 and 6 were all price-only.
3. **Macro events condition the micro signal**: a daily market-level event
   vector FiLM-transforms each ticker's own text representation before the
   graph mixes tickers.

The question: does a graph built from what people say, whose per-ticker signal
is read in light of what is happening to the market, carry signal that sector
cliques and price correlation do not?

**The answer is no, and the controls say so decisively.**

## 2. The MAN-SF non-reproduction caveat

None of the graphs built here is MAN-SF's graph. Their reported GAT+price MCC
of 0.117 and headline 0.195 were measured on a **Wikidata company-relation
graph**, which is not built in this project. No result in this document may be
described as reproducing, approaching, or beating those figures.

## 3. Task 0 — the `Price3` fix, and its effect on Experiment 6

Measured across all 106,476 day-slots of the protocol-A parquet, the third
price feature was degenerate:

| Population | `Price3` std | `Price3` range |
|---|---|---|
| All 87 tickers | **154.00** | [−6495, +6500] |
| Excluding BRK-A and PCLN | **1.41** | [−44.3, +93.1] |

Only **255 rows of 26,619** exceed \|100\|: **BRK-A (252)** and **PCLN (3)** —
the two highest-priced tickers in the universe. Because
`normalize_price_snapshots` computes one global mean/std, those two inflated
std by 109× and crushed every other ticker's signal. Measured on the real
pipeline, for all tickers except BRK-A and PCLN:

| | post-z `Price3` std | post-z `Price3` range |
|---|---|---|
| unfixed | **0.0103** | [−0.330, +0.677] |
| winsorized at ±10 | **0.7550** | [−6.674, +6.679] |

A **73× recovery of dynamic range**. At std 0.0103 a third of the node features
was effectively a constant zero — in Experiments 3, 4, 5 and 6.

Fixed by winsorizing at ±10 before normalization (a fixed limit, not a
data-derived percentile, so it cannot leak between splits). BRK-A is a
legitimate universe member and was not dropped — dropping it would move test
`n_samples` off 3720 and break comparability.

**Effect on Experiment 6:** re-running its `GAT [sector] seed=0` cell with the
fix moved that cell from **+0.0263 to +0.0435**. This is a **single seed**, and
Experiment 6 measured a 3-seed spread of ~0.029 for that cell, so the shift sits
inside noise and **does not by itself establish an improvement**. Experiment 6's
conclusions are not overturned by this, but every experiment before this one ran
on two usable price features rather than three, and that should be stated
whenever their numbers are cited.

## 4. The between-day / within-day problem

Measured variance decomposition of the protocol-A target:

| Split | Days | Between-day variance | Days >80% one-directional | Market-direction oracle |
|---|---|---|---|---|
| train | 398 | 40.7% | 167/398 | acc 0.7811, MCC +0.5622 |
| dev | 42 | 56.8% | 24/42 | acc 0.8454, MCC +0.6863 |
| **test** | **64** | **48.3%** | **32/64** | **acc 0.8116, MCC +0.6248** |

Roughly half the label variance is *market direction*, not *which stock
outperforms which*. A predictor that knows only the day's direction and cannot
tell stocks apart scores pooled **MCC +0.6248** — far above every published
number on this benchmark (StockNet 0.081, MAN-SF 0.195).

Pooled MCC therefore conflates two abilities. The graph exists to do the second
one. **Every row below is reported on both metrics**, and the honest reading
leads with within-day.

## 5. Data, as measured

FNSPID (`Stock_news/All_external.csv`, 5.34 GB, arXiv 2402.06698), streamed once
and filtered to 2014-2015, English only, split into a market-wide macro channel
(no `Stock_symbol`) and a per-ticker micro channel.

| Channel | Headlines | Day coverage | Per day |
|---|---|---|---|
| Macro | 1,782,272 | **503/504 (99.8%)** | median 3,505 |
| Micro | 18,785 | 504/504 (100%) | median 34 |

The spec estimated ~3.9M macro headlines at ~7,600/day from a clustered
byte-probe; the measured figures are 1.78M at ~3,505/day.

**A per-day cap of 64 was mandatory, not an optimisation.** Uncapped, the macro
channel alone is ~12 hours of CPU FinBERT. Capped, the whole embedding job was
138,642 vectors in about two hours.

**The micro news channel is thin and systematically biased** — this materially
limits what row 2 onward can claim:
- only **52 of 87 tickers have any news at all**; 35 have none, ever
- only **8,836 of 43,848 (ticker, trading-day) cells** are non-empty = **20.2%**

So the micro signal is dominated by **tweets**; the news half is zero for ~80%
of nodes. Any signal attributed to the news channel is partly encoding *which
tickers get press coverage*, which tracks size and liquidity rather than market
timing.

**Tweet graph density:** mean **211.1 edges/day**, max 325 — a real object,
denser than Experiment 6's co-mention relation (10.8/day, its worst) and
comparable to its correlation relation (321/day).

**But a large share of nodes are isolated in it.** A ticker with no tweets in
its `t−1…t−5` window gets a zero tweet vector and is excluded from the graph
entirely. Measured: **1,567 of 3,720 test samples (42.1%)** — and 11,445 of
26,619 overall (43.0%) — have an empty tweet window. For those nodes the graph
rows are functionally identical to the no-graph row, which is the strongest
single caveat on any graph conclusion below.

## 6. Results — the ablation ladder

All rows: 5 seeds, mean ± std. `n_samples = 3720` on every one of the 40 runs.
Within-day MCC is computed per trading-day cross-section and averaged; 63 of 64
test days were scored (1 skipped as single-class).

| # | Row | Within-day MCC | Pooled MCC | AUC |
|---|---|---|---|---|
| 6 | attention-site conditioning | **+0.0183 ± 0.0095** | +0.0145 ± 0.0228 | 0.5027 |
| 5 | **shuffled-macro control** | **+0.0173 ± 0.0200** | +0.0127 ± 0.0237 | 0.5241 |
| 3 | tweet graph, no conditioning | +0.0169 ± 0.0252 | +0.0153 ± 0.0120 | 0.5076 |
| 4 | **full model (macro→micro FiLM)** | **+0.0127 ± 0.0235** | −0.0005 ± 0.0151 | 0.5208 |
| 2 | micro features, no graph | +0.0105 ± 0.0028 | +0.0000 ± 0.0199 | 0.5065 |
| 7 | sector graph + conditioning | +0.0065 ± 0.0339 | +0.0213 ± 0.0138 | 0.5275 |
| 0 | **macro-only control** | **+0.0000 ± 0.0000** | −0.0388 ± 0.0354 | 0.4926 |
| 1 | price-only | −0.0053 ± 0.0126 | +0.0044 ± 0.0154 | 0.5135 |

Reference points (pooled MCC): Experiment 1 LSTM+tweets **+0.0922**;
Experiment 6 best (RGCN sector+correlation) **+0.0677 ± 0.0255**; Experiment 6
GAT sector **+0.0160 ± 0.0161**; always-UP constant acc 0.5129 / F1 0.6780;
always-DOWN acc 0.4871 / F1 0.0; **market-direction oracle pooled MCC +0.6248 /
acc 0.8116**.

### The control that validates the measurement

Row 0 returned within-day MCC of **exactly +0.0000 on all five seeds**. That is
required by construction — it has no per-ticker input, so every ticker on a day
receives an identical prediction. Both the within-day metric and the model's
`macro_only` short-circuit are therefore behaving as designed.

## 7. Honest reading

**Macro conditioning does not work. The decisive control fails outright.**

| Comparison | Difference | p |
|---|---|---|
| full (row 4) vs **shuffled-macro (row 5)** | **−0.0046** | 0.747 |
| full (row 4) vs graph-only (row 3) | −0.0043 | 0.788 |
| full (row 4) vs macro-only (row 0) | +0.0127 | 0.262 |

Row 5 re-runs the full model with macro vectors **randomly permuted across the
calendar** — identical architecture, identical parameter count, destroyed
event-alignment. It scores **higher** than the real model. Whatever row 4
achieves is therefore attributable to added capacity, not to macro events. Under
the spec's own stated criterion the headline claim is **withdrawn**.

Row 4 also fails to beat row 3, so conditioning adds nothing over the tweet
graph alone.

**The explainability readout is consistent with that.** In the seed examined,
the learned FiLM parameters barely move across the test period: `‖γ‖` has a
coefficient of variation of **2.3%** and `‖β‖` **3.7%** — the model has
effectively learned to *ignore* the macro channel. There is a nominally
significant lift on macro-event days (`‖γ‖` 0.6616 on the 8 event days vs
0.6503 otherwise, **p = 0.049**; `‖β‖` p = 0.111), but the effect is ~1.7%
relative on 8 days and is what a near-constant transform would produce anyway.

**Caveat on this readout:** it covers the 64 test days of **one seed** (seed 3),
chosen by `nlargest(1, "mcc")` — that is, the seed with the highest *test*
pooled MCC. Selecting on test is a real limitation and the claim is scoped to
that seed accordingly. The direction is benign for the headline: if anything,
test-selection favours the seed most likely to *show* conditioning, and it still
shows a near-constant transform. But "the model learned to ignore macro" is
established for the examined seed, not for all five.

**Does anything beat price-only?** Marginally on within-day MCC, and the AUC
column contradicts it.

| Row vs price-only | Within-day diff | p | That row's AUC | vs price-only AUC 0.5135 |
|---|---|---|---|---|
| 6 attention-site | +0.0236 | 0.010 | 0.5027 | **−0.0108** |
| 3 graph, no conditioning | +0.0223 | 0.116 | 0.5076 | **−0.0059** |
| 4 full | +0.0180 | 0.170 | 0.5208 | +0.0073 |
| 2 micro, no graph | +0.0158 | 0.026 | 0.5065 | **−0.0070** |
| 7 sector graph | +0.0118 | 0.487 | 0.5275 | +0.0140 |

All p-values are unpaired, equal-variance two-sample t-tests on 5 seeds vs 5.

Two rows clear p < 0.05 nominally. Three things undercut them:

1. **AUC disagrees.** The two rows with the *best* within-day MCC — 6 and 3 —
   are *below* price-only on AUC, and row 6, the ladder's top within-day row,
   sits at 0.5027, essentially chance. That is the threshold-luck pattern this
   project has been guarding against since Experiment 6, now appearing on the
   within-day metric rather than the pooled one.
2. **The noise floor exceeds every effect.** Mean within-row seed spread is
   **0.0387**; the largest between-row gap is **0.0236**.
3. **These are 5 comparisons on 5 seeds each**, so nominal significance does not
   survive correction robustly. (A paired test, arguably more natural since the
   seeds are shared, moves row 6 to p = 0.023 and row 3 to p = 0.103 — same
   conclusion.)

By the rule this project has applied since Experiment 6, that is
*indistinguishable*, not an improvement.

**For scale:** a perfect per-day ranker — one that orders every day's
cross-section correctly — scores within-day MCC **+0.5152** on this test split.
The best row in the ladder is therefore **3.6% of achievable**, and the seed
noise floor alone is **7.5%** of it.

**On pooled MCC, Experiment 7 underperforms its predecessors.** The best row
mean is **+0.0213** (row 7), against Experiment 6's best **+0.0677** and
Experiment 1's **+0.0922**. The nearest like-for-like comparison — row 7
(sector graph) at +0.0213 pooled versus Experiment 6's GAT sector at
+0.0160 ± 0.0161 — is a wash.

**Summary:** the learned tweet graph is a real, well-formed object, and putting
text on the nodes is at best marginally better than price alone. Conditioning
that signal on macro events contributes nothing measurable, and the shuffled
control proves it.

## 8. Leakage audit

The spec's tripwire — any cell above pooled MCC 0.15 is treated as suspected
leakage rather than success — **did not fire**. Maximum pooled MCC across all
40 runs is **+0.0410**.

Leakage is a tested property throughout, on three separate cutoffs:
- macro headlines must precede day *t*'s 09:30 ET open;
- micro news must precede the same open;
- tweets use a window ending at *t−1*, excluding the target day.

Each has a test that places an item at **exactly** the boundary instant and
asserts exclusion, so a `<=` off-by-one would fail rather than pass silently.
The full suite is 128 tests.

**One nuance worth stating rather than glossing.** The label is StockNet's
`Main_MV_Percent`, a close-to-close movement for the target date, which already
contains the overnight gap from close(t−1) to open(t). Pre-open news on day *t*
is therefore contemporaneous with part of the label rather than strictly
predictive of all of it. The 09:30 cutoff is exactly what the spec mandates and
is implemented correctly, so this is a property of the task definition, not a
leak — and since the result is null it could only have flattered the model.

## 9. Disclosure — weaknesses and uncontrolled differences

1. **Training stops very early — more often than in Experiment 6.**
   `best_epoch` row means span **2.8 to 17.0** (the floor is row 0), and
   **12 of 40 cells (30%) stopped below epoch 3**, five of them at epoch 0.
   Those cells describe a barely-trained model. Experiment 6 had 14 of 84
   (17%), so Experiment 7 early-stops roughly **twice as often** — this is
   worse than its predecessor, not comparable to it. The cause is patience-10
   early stopping against a 42-day dev set.
2. **Model selection uses the protocol dev set**, not a train-carved fold. This
   was a deliberate choice to keep row 1 comparable with Experiment 6's
   price-only control, at the cost of inheriting that 42-day set's selection
   noise — a plausible contributor to the 0.0387 seed spread.
3. **CPI and NFP calendar dates are derived by convention** (NFP = first Friday;
   CPI = the 15th or following Monday) and are **not verified against the BLS
   release archive**; they may be off by a day or two. FOMC dates are exact.
4. **The micro news channel covers only 52 of 87 tickers and 20.2% of
   (ticker, trading-day) cells** (see §5). Conclusions about "news" are weak by
   construction.
5. **Per-day cap of 64** bounds macro fidelity. The cap applies per calendar day
   *within* the 3-day lookback, so each macro vector averages **255.5 headlines
   on average** (median 256, min 128, max 256; 128,750 in total) out of a median
   3,505 available per day, selected by recency.
6. **Two different, often-confused figures on tweet coverage.** Exactly **17 of
   26,619** protocol-A `(Ticker, Target_Date)` pairs are missing from the
   modeling set entirely. Separately and far more consequentially,
   **1,567 of 3,720 test samples (42.1%)** have no tweet text anywhere in
   `t−1…t−5`; they receive a zero tweet vector, are excluded from the tweet
   graph, and reach the GAT with only a self-loop. For those ~42% of test nodes
   the graph rows are functionally the no-graph row.
7. **Shared FiLM parameters.** `γ` and `β` are identical across every ticker on
   a day, so macro conditioning cannot by itself reorder tickers within a day —
   only rescale their micro signal before the graph mixes them. This is a real
   structural limit on what row 4 could ever have achieved, and given the null
   result it is the leading explanation.
8. **The FNSPID parser anchors rows on a leading timestamp.** An article-body
   continuation line that itself begins with a timestamp-shaped string would be
   misparsed as one garbage row with an empty symbol, landing in the macro
   channel. Bounded to a handful of rows among 1.78M.
9. **Non-English filtering is an ASCII-ratio heuristic**, not a language model,
   and its **reject rate was not recorded** — the spec asked for it and the
   filter drops rows without a counter, so recovering the rate needs a re-stream.
10. **A small tweet-embedding cache mismatch.** The cache stores each
    (ticker, day) window's *last* 64 tweet segments while the encoder requests
    the *first* 64 of a most-recent-first ordering. On the 70 of 12,929 keys
    that exceed the cap these sets partly disjoin, so **2,368 of 192,947
    requested tweet texts (1.23%)** are cache misses and are silently dropped
    from the mean. No window is fully lost. Macro and micro news were verified
    at 100% cache coverage before the run; the tweet channel was not, and should
    be included in any future pre-flight check.
11. **Two spec deliverables in Component 7 were not produced**: ranking days by
    \|γ − mean γ\| with their top headlines, and comparing the learned tweet
    graph's edges against the sector and correlation graphs. The second is the
    evidence that would have substantiated any claim that the tweet graph is a
    better substrate, and its absence is why §10 makes no such claim.

## 10. How we should move forward

1. **Do not pursue macro→micro FiLM conditioning further.** The shuffled control
   settles it. If conditioning is revisited, the shared-`γ` limitation in §9.7 is
   the thing to fix first — a mechanism that cannot reorder tickers within a day
   cannot help a within-day metric except through interaction effects.
2. **Fix the evaluation before adding more capacity.** The seed spread (0.0387)
   exceeds every effect measured here (max 0.0236). Until selection noise is
   reduced — walk-forward validation, or selection folds carved from train — no
   architectural change in this project can be resolved. This is now the third
   consecutive experiment where the noise floor has swamped the effect.
3. **The learned tweet graph is not yet shown to be a better substrate, and
   this experiment cannot settle it.** It is non-degenerate at 211 edges/day and
   rows 3/6 are the best within-day rows — but three things block any stronger
   claim: those same rows are *below* price-only on AUC (§7); **42% of test
   nodes are isolated in the graph** for want of tweets (§5); and the
   spec-mandated comparison of its edges against the sector and correlation
   graphs was never run (§9.11). Running that overlap analysis, and measuring
   performance restricted to the 58% of nodes that actually have neighbours,
   are the two cheapest experiments that would answer this properly.
4. **Text on nodes is the most promising remaining direction**, consistent with
   Experiment 1 (+0.0922 with tweets) being the project's best pooled result.
   Every graph experiment so far — 4, 5, 6, and 7's price-only row — has
   underperformed it.
5. **Report within-day metrics from here on.** Pooled MCC on this dataset
   rewards market-direction learning that no graph is responsible for, and the
   oracle at +0.6248 shows how much room there is to be misled.
