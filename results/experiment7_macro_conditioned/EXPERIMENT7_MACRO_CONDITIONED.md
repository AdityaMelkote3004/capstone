# Experiment 7 — Macro-Conditioned Micro Signal over a Tweet Graph

**Status:** Complete. 45/45 ladder cells trained (9 rows × 5 seeds).
**Run:** local CPU (PyTorch 2.11.0+cpu, no GPU), `scripts/run_experiment7_ladder.py`.
**Raw results:** `results.csv` (45 rows) + one `metrics.json` per run.
**Analysis:** `scripts/analyze_experiment7.py` regenerates every derived number below.
**Spec:** `docs/superpowers/specs/2026-09-08-event-conditioned-tweet-graph-design.md`
**Superseded measurement:** `v1_lookback5/` — see §6, which documents the protocol
change and reports both versions rather than replacing one with the other.

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

**The answer splits.** Putting text on the nodes and running a graph over it
produces the first effect in this project that clears its own noise floor:
+0.0425 within-day MCC over price-only, p < 0.0001, surviving Bonferroni
correction, corroborated by AUC, positive on all five seeds. **Macro
conditioning contributes nothing** — it is statistically indistinguishable from
conditioning on *shuffled* macro vectors (+0.0002, p = 0.987). And the claim
that the *tweet-learned* graph specifically beats sector cliques is **not
supported** at this sample size (§7.3).

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
channel alone is ~12 hours of CPU FinBERT. The embedding cache holds **139,624
vectors**: 69,072 unique macro headlines (drawn by 128,750 windowed requests —
the two figures differ because overlapping 3-day windows request the same
headline on consecutive days), 18,785 micro headlines, and the tweet segments.

**The micro news channel is thin and systematically biased** — this materially
limits what row 2 onward can claim:
- only **52 of 87 tickers have any news at all**; 35 have none, ever
- only **8,836 of 43,848 (ticker, trading-day) cells** are non-empty = **20.2%**

So the micro signal is dominated by **tweets**; the news half is zero for ~80%
of nodes. Any signal attributed to the news channel is partly encoding *which
tickers get press coverage*, which tracks size and liquidity rather than market
timing.

**Tweet graph density:** mean **318.0 edges/day**, max 365 across all splits
(321.9/day on test days) — comparable to the sector graph's 381 static edges and
the correlation graph's 339.7/day, so the three are compared at similar density.

**Node isolation, the constraint that drove the v2 re-run.** A ticker with no
tweets anywhere in its lookback window gets a zero tweet vector and is excluded
from the graph entirely; for that node, every graph row is functionally the
no-graph row. At v1's 5-day lookback this affected **1,567 of 3,720 test samples
(42.1%)**. At v2's 20-day lookback it affects **658 of 3,720 (17.7%)**
— 4,418 of 26,619 (16.6%) across all splits. This is still the strongest single
caveat on any graph conclusion below.

## 6. The v1 → v2 protocol change, and why it is not tuning

The ladder was run twice. The first run (`v1_lookback5/`, 40 cells) is a real
measurement and is not withdrawn; it is an **under-powered** one, and this
section says exactly why, what was changed, and what evidence distinguishes a
principled fix from fitting the protocol to the answer.

### 6.1 What changed

Three defects, all identified and fixed **before any v2 number was seen**:

| # | Defect in v1 | Fix | Measured effect on the pipeline |
|---|---|---|---|
| 1 | 5-day tweet lookback left **42.1% of test nodes isolated** from the graph, so the graph rows were largely untestable | lookback 5 → 20 | isolation **42.1% → 17.7%** |
| 2 | patience-10 against a 42-day dev set: **12 of 40 cells stopped below epoch 3**, five at epoch 0 — those cells reported a barely-trained model | `MIN_EPOCHS=15`, patience 10 → 20 | mean `best_epoch` **10.0 → 20.3**; cells below epoch 3 **12/40 → 5/40** |
| 3 | the cache stored each window's **last** 64 tweet segments while the encoder requests the **first** 64 of a most-recent-first ordering, silently dropping 1.23% of requested texts | `segments[:CAP]` | all three channels verified at **100%** cache hit rate (tweets: 607,566 requested, 0 misses) |

Defect 1 is the one that matters. It was not a hyperparameter: a node excluded
from the graph makes the graph rows *identical to the no-graph row* for that
node, so at 42% isolation the experiment could not measure the thing it existed
to measure.

### 6.2 The evidence that this is a fix, not tuning

A protocol change that moves the headline result upward deserves suspicion. The
test is whether the change moved results **where its mechanism predicts** and
left everything else alone. It did:

| Row | v1 | v2 | Δ | has a graph? |
|---|---|---|---|---|
| 0 macro-only (control) | +0.0000 | +0.0000 | **0.0000** | no |
| 1 price-only | −0.0053 | +0.0010 | +0.0063 | no |
| 2 micro features, no graph | +0.0105 | +0.0131 | +0.0025 | no |
| 3 tweet graph, no conditioning | +0.0169 | +0.0436 | **+0.0266** | yes |
| 4 full model | +0.0127 | +0.0282 | +0.0155 | yes |
| 5 shuffled-macro control | +0.0173 | +0.0280 | +0.0108 | yes |
| 6 attention-site conditioning | +0.0183 | +0.0338 | +0.0155 | yes |
| 7 sector graph + conditioning | +0.0065 | +0.0160 | +0.0096 | yes |

Three things follow, and together they are hard to get from tuning:

1. **The graph rows moved ~5× more than the non-graph rows** (mean Δ +0.0156 vs
   +0.0029). The isolation fix acts *only* through graph membership, so this is
   its signature.
2. **Row 2 is the decisive control.** It consumes the same widened 20-day tweet
   features as row 3 but has no graph — and it moved only **+0.0025**. So
   widening the window did almost nothing to the *features*; the gain is
   specifically in the *graph*. A change that simply gave every row more text
   would have moved row 2 as much as row 3.
3. **Row 0 moved exactly 0.0000 on all five seeds.** It has no per-ticker input,
   so it cannot move — and it didn't, in either version.

**The strongest evidence is what did *not* improve.** The conditioning contrast
(row 4 − row 5) went from −0.0046 in v1 to +0.0002 in v2. Three protocol changes
were made and the control that was supposed to stay null **stayed null**. Tuning
toward a better story would not have left the paper's own headline mechanism —
macro conditioning — dead in the water.

### 6.3 What a reader should still hold against it

The correction family in §7.2 is **declared post hoc**: the spec pre-registered
the ablation ladder but not a multiple-comparison correction, so the family of
seven contrasts is author-chosen, fixed before its p-values were read but not
before the experiment. n = 5 seeds is a small sample for a t-test. And the
decision to re-run at all was taken *after* seeing v1's null — the fixes are
independently justified, but the trigger to look for them was a disappointing
result, which is the classic setting for motivated protocol change. §6.2 is
offered as the check on that, not as a dismissal of it.

## 7. Results — the ablation ladder

All rows: 5 seeds, mean ± std. `n_samples = 3720` on every one of the 45 runs.
Within-day MCC is computed per trading-day cross-section (thresholded at that
day's median) and averaged; 63 of 64 test days were scored (1 skipped as
single-class).

| # | Row | Within-day MCC | Pooled MCC | AUC | `best_epoch` |
|---|---|---|---|---|---|
| 3 | **tweet graph, no conditioning** | **+0.0436 ± 0.0072** | +0.0412 | 0.5322 | 20.8 |
| 8 | **sector graph, no conditioning** | **+0.0350 ± 0.0141** | +0.0245 | 0.5360 | 23.2 |
| 6 | attention-site conditioning | +0.0338 ± 0.0062 | +0.0273 | 0.5209 | 37.2 |
| 4 | full model (macro→micro FiLM) | +0.0282 ± 0.0154 | +0.0459 | 0.5388 | 22.0 |
| 5 | **shuffled-macro control** | +0.0280 ± 0.0122 | +0.0393 | 0.5392 | 14.8 |
| 7 | sector graph + conditioning | +0.0160 ± 0.0322 | +0.0405 | 0.5378 | 14.2 |
| 2 | micro features, no graph | +0.0131 ± 0.0163 | +0.0360 | 0.5236 | 22.8 |
| 1 | price-only | +0.0010 ± 0.0070 | −0.0013 | 0.5236 | 16.0 |
| 0 | **macro-only control** | +0.0000 ± 0.0000 | −0.0714 | 0.4889 | 11.2 |

Reference points (pooled MCC): Experiment 1 LSTM+tweets **+0.0922**;
Experiment 6 best (RGCN sector+correlation) **+0.0677 ± 0.0255**; Experiment 6
GAT sector **+0.0160 ± 0.0161**; always-UP constant acc 0.5129 / F1 0.6780;
**market-direction oracle pooled MCC +0.6248 / acc 0.8116**.

**Noise floor (mean seed spread across rows): 0.0304.** Any difference smaller
than this is not resolvable by this experiment.

**Tripwire:** max pooled MCC across all 45 cells is **+0.0621**, well below the
0.15 threshold that would have indicated leakage. It did not fire.

### 7.1 The controls behaved

- **Row 0** returned within-day MCC of **exactly +0.0000 on all five seeds**.
  Required by construction — it has no per-ticker input, so every ticker on a
  day gets an identical prediction and the within-day ranking is degenerate.
  Both the metric and the model's `macro_only` short-circuit work as designed.
- **Row 5** (real architecture, macro vectors randomly permuted across days)
  scored **+0.0280**, statistically identical to row 4's +0.0282.

### 7.2 Declared contrasts

Bonferroni over the **seven** contrasts the ladder was built to make,
α = 0.05/7 = **0.0071**. This family is **declared post hoc** (see §6.3).

| Contrast | Diff | Unpaired p | Paired p | > noise? | Survives? | AUC |
|---|---|---|---|---|---|---|
| **graph vs price-only** (3−1) | **+0.0425** | **<0.0001** | 0.0013 | **yes** | **YES** | **agrees, p=0.021** |
| graph vs micro-no-graph (3−2) | +0.0305 | 0.0102 | 0.0378 | yes | no | agrees, p=0.143 |
| micro vs price-only (2−1) | +0.0120 | 0.1849 | 0.0987 | no | no | disagrees |
| tweet graph vs sector (3−8) | +0.0086 | 0.2690 | 0.3185 | no | no | **disagrees** |
| conditioning vs shuffled (4−5) | **+0.0002** | **0.9866** | 0.9870 | no | no | disagrees |
| conditioning vs none (4−3) | −0.0154 | 0.0927 | 0.0638 | no | no | disagrees |
| attention site vs FiLM (6−3) | −0.0098 | 0.0512 | 0.0404 | no | no | agrees |

### 7.3 The three findings

**1. A graph over text-bearing nodes beats price-only. This one holds.**

Row 3 vs row 1 is +0.0425 with unpaired p < 0.0001 and paired p = 0.0013. It
**survives Bonferroni**, it **exceeds the 0.0304 noise floor** — the first
effect in this project to do so — **AUC independently agrees** (0.5322 vs
0.5236, p = 0.021), and it is **positive on all five seeds**
(+0.0420 / +0.0387 / +0.0469 / +0.0360 / +0.0542) against a price-only row that
is positive on only two.

Row 8 replicates it with a completely different graph: +0.0350, also 5/5
positive. Pooling the two unconditioned graph rows against price-only gives
**+0.0382, p = 0.000004, 10/10 seeds positive**.

**2. The learned tweet topology is *not* the source of the gain.**

This is the claim the experiment was designed to make, and it fails. The tweet
graph beats the sector graph by **+0.0086 — p = 0.269, far below the noise
floor, and AUC points the other way** (sector 0.5360 vs tweet 0.5322). At five
seeds these two graphs are indistinguishable.

That is not because the graphs are secretly the same object. Measured over the
64 test days (`tweet_graph_overlap.json`):

| | Tweet graph | vs sector | vs correlation |
|---|---|---|---|
| edges/day | 321.9 | 381 (static) | 339.7 |
| Jaccard | — | **0.084** | **0.062** |
| random-graph baseline at these densities | — | 0.049 | 0.046 |
| share of tweet edges also present | — | **17.0%** | 12.0% |

The tweet graph is only ~1.7× more sector-aligned than a random graph of the
same density; **83% of its edges are not sector edges**. So two structurally
very different graphs deliver statistically the same benefit. The natural
reading is that what helps is **message passing over text features at all**,
not the particular topology — which is a weaker claim than the experiment set
out to make, and the honest one.

**3. Macro conditioning contributes nothing.**

Row 4 vs row 5 is **+0.0002 (p = 0.987)**. Conditioning on the real macro
stream is indistinguishable from conditioning on macro vectors shuffled across
days. The conditioning claim is withdrawn.

Directionally, removing conditioning *helps* on both graphs — tweet +0.0154
(p = 0.093), sector +0.0189 (p = 0.279), pooled +0.0172 (p = 0.068) — but
**AUC disagrees in sign on both**, so "conditioning actively hurts" is
**not established**; only "conditioning does not help" is.

**The mechanism is visible in the model.** The FiLM readout (`film_by_day.csv`,
row 4 seed 2, 64 test days) shows γ and β that barely move as the macro vector
changes. The conditioning generator learned to emit a near-constant transform —
the network **learned to ignore the macro channel** rather than being defeated
by it, which is consistent with the shuffled control scoring identically.

The readout was extracted three times on the same configuration, and it is only
qualitatively stable:

| Extraction | ‖γ‖ mean | ‖γ‖ CV | ‖β‖ CV |
|---|---|---|---|
| v1 (lookback 5) | 0.6517 | 2.33% | 3.65% |
| v2, during the 8-row ladder | 0.7113 | 1.48% | 2.66% |
| v2, re-extracted with row 8 | 0.4274 | **0.00%** | **0.00%** |

In the third extraction ‖γ‖ is **bit-identical across all 64 test days** — one
distinct value — meaning the generator collapsed completely and the macro input
changes nothing at all. All three agree that conditioning is inert; **the exact
CV should not be quoted as a stable measurement**, because the three runs used
the same seed and the same data and still differ (see §9.13).

### 7.4 How big is +0.0425, really?

Small. A perfect per-day ranker — one that orders every day's cross-section
correctly — scores within-day MCC **+0.5152** on this test split. Row 3's
+0.0436 is **8.5% of that ceiling**.

On pooled MCC, row 4's +0.0459 and row 3's +0.0412 remain **below Experiment 1's
LSTM+tweets (+0.0922)** and below Experiment 6's best RGCN (+0.0677). Nothing
here beats the project's own earlier pooled numbers, and nothing here approaches
published benchmarks. What is new is that this is the first effect measured in
this project that is **larger than the noise it is measured against** — the
finding is about *resolvability*, not about magnitude.

## 8. Leakage audit

The spec's tripwire — any cell above pooled MCC 0.15 is treated as suspected
leakage rather than success — **did not fire**. Maximum pooled MCC across all
45 runs is **+0.0621**.

Leakage is a tested property throughout, on three separate cutoffs:
- macro headlines must precede day *t*'s 09:30 ET open;
- micro news must precede the same open;
- tweets use a window ending at *t−1*, excluding the target day.

Each has a test that places an item at **exactly** the boundary instant and
asserts exclusion, so a `<=` off-by-one would fail rather than pass silently.
The full suite is **128 tests, all passing** under the v2 configuration.

**The overnight-gap nuance, which now matters more than it did.** The label is
StockNet's `Main_MV_Percent`, a close-to-close movement for the target date,
which already contains the overnight gap from close(t−1) to open(t). Pre-open
news on day *t* is therefore **contemporaneous with part of the label** rather
than strictly predictive of all of it. The 09:30 cutoff is exactly what the spec
mandates and is implemented correctly, so this is a property of the task
definition rather than an implementation leak.

In v1 this could be waved off because the result was null. **It cannot be now.**
The node features in row 3 concatenate a tweet vector (clean: strictly ≤ *t−1*)
with a news vector (pre-open day *t*, hence partly contemporaneous). The tweet
*graph* is built from the clean half only, but message passing then mixes the
news half across neighbours. So the headline +0.0425 has an untested
alternative explanation: that the graph is propagating overnight-gap information
rather than predictive signal.

Two things bound this, but neither settles it: the news channel is empty for
**79.8%** of (ticker, day) cells, and row 2 — the same features without a graph
— is only +0.0131 (n.s.). **The decisive test is row 3 re-run with the news half
zeroed**, which is the first item in §10 and was not run for this document.

## 9. Disclosure — weaknesses and uncontrolled differences

1. **The headline finding has an unrun control** — the news-half ablation in §8.
   Until it exists, "the graph adds predictive signal" and "the graph propagates
   overnight-gap information" are not separated.
2. **The experiment's own thesis is unsupported.** The tweet-learned graph does
   not beat sector cliques (§7.3). Experiment 7 was designed to show that a
   graph learned from tweet semantics is a better substrate; it did not.
3. **n = 5 seeds** is a small sample for a t-test, and the Bonferroni family is
   **declared post hoc** (§6.3). The re-run was triggered by a disappointing v1.
4. **Model selection uses the protocol dev set** (42 days), not a train-carved
   fold, inheriting that set's selection noise — a plausible contributor to the
   0.0304 seed spread.
5. **Still 17.7% of test nodes are isolated** from the tweet graph (§5). For
   those the graph rows remain functionally the no-graph row, which biases every
   graph contrast *downward*.
6. **The micro news channel covers only 52 of 87 tickers and 20.2% of
   (ticker, trading-day) cells.** Conclusions about "news" are weak by
   construction, and partly encode which tickers get press coverage.
7. **Per-day cap of 64** bounds macro fidelity: each macro vector averages
   **255.5 headlines** (median 256, min 128, max 256; 128,750 windowed requests
   over 69,072 unique headlines) out of a median 3,505 available per day,
   selected by recency.
8. **CPI and NFP calendar dates are derived by convention** (NFP = first Friday;
   CPI = the 15th or following Monday) and are **not verified against the BLS
   release archive**. FOMC dates are exact.
9. **Shared FiLM parameters.** γ and β are identical across every ticker on a
   day, so macro conditioning cannot by itself reorder tickers within a day —
   only rescale their micro signal before the graph mixes them. Given the null
   result and the near-constant γ/β (§7.3), this remains the leading structural
   explanation.
10. **The FNSPID parser anchors rows on a leading timestamp.** An article-body
    continuation line beginning with a timestamp-shaped string would be
    misparsed as one garbage row with an empty symbol, landing in the macro
    channel. Bounded to a handful of rows among 1.78M.
11. **Non-English filtering is an ASCII-ratio heuristic**, not a language model,
    and its **reject rate was not recorded** — recovering it needs a re-stream.
12. **One spec deliverable remains unproduced**: ranking days by \|γ − mean γ\|
    with their top headlines. In the latest extraction γ is *constant to the
    bit*, so the ranking would be undefined rather than merely noisy — but it
    was specified and is absent.
13. **Training is not bit-reproducible run to run.** Re-training row 4 at the
    same seed, on the same data, with the same code produced a materially
    different FiLM readout (§7.3). `set_seed` covers Python, NumPy and torch
    RNG but the run does not set `torch.use_deterministic_algorithms(True)` or
    pin thread count, and torch-geometric's CPU scatter accumulates in
    nondeterministic order. So the seed spread reported here bundles
    initialisation variance *and* run-to-run numerical variance, and single-cell
    numbers should be treated as draws rather than fixed values. The five-seed
    contrasts in §7.2 are unaffected in kind — they compare distributions — but
    this should be fixed before any result is quoted at more precision than the
    noise floor.

## 10. How we should move forward

1. **Run the news-half ablation before citing +0.0425 anywhere.** Re-run rows 1,
   2, 3 and 8 with the news vector zeroed, leaving only tweet text (strictly
   ≤ *t−1*). If the graph advantage survives, the finding is clean and
   defensible. If it collapses, the effect was overnight-gap propagation and
   must be reported as such. This is 20 cells, ~2 hours of CPU, and it is the
   single highest-value thing left in this experiment.
2. **Do not pursue macro→micro FiLM conditioning further.** The shuffled control
   settles it, and the near-constant γ/β explains why. If it is ever revisited,
   fix the shared-γ limitation in §9.9 first — a mechanism that cannot reorder
   tickers within a day cannot help a within-day metric except through
   interaction effects.
3. **Drop the "learned graph is a better substrate" thesis, or test it properly.**
   Two structurally dissimilar graphs (17% edge overlap) scored the same. Either
   the topology genuinely does not matter here — in which case the interesting
   question is why message passing over text helps at all — or five seeds cannot
   resolve a real difference. Both readings point to more seeds, not more
   architecture.
4. **Fix the evaluation before adding capacity.** The noise floor (0.0304) still
   exceeds most effects measured here. Walk-forward validation, or selection
   folds carved from train, would buy more than any new module. This experiment
   is the first in the project where an effect cleared that floor — the way to
   get more such results is to lower the floor.
5. **Report within-day, always.** Pooled MCC on this benchmark is dominated by
   market direction; the oracle at +0.6248 shows how much room there is to be
   misled by it.
