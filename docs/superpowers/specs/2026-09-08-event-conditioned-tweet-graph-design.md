# Experiment 7 — Macro-Conditioned Micro Signal over a Tweet Graph — Design Spec

**Status:** design approved 2026-09-08, revised same day, not yet implemented.
**Predecessor:** `2026-09-03-dense-graph-experiment-design.md` (Experiment 6).

## Purpose

Experiment 6 measured every combination of hand-designed edges (sector,
rolling correlation, tweet co-mention) across four graph models and three
seeds, and found **no distinguishable improvement** over a sector-only
graph: mean within-cell seed spread was 0.0247, larger than the gap between
the first- and third-ranked cells, and 8 of 28 (model, edge) cells sat at
AUC below 0.51.

Experiment 7 changes three things Experiment 6 held fixed:

1. **The graph is learned from tweet semantics** rather than hand-designed
   from sector membership or price correlation.
2. **Node features carry text.** Experiments 4, 5, and 6 were all
   price-only — a fact that materially weakens their negative findings
   about graphs, and which is stated as such in the write-up.
3. **Macro events condition the micro signal.** A daily market-level
   event vector modulates each ticker's own text representation, so the
   same headline or tweet is read differently against a different
   macroeconomic backdrop.

The question it answers: *does a graph built from what people say, whose
per-ticker signal is interpreted in light of what is happening to the
market as a whole, carry signal that sector cliques and price correlation
do not?*

### Why the previous graphs plausibly failed

- **Sector edges** are static, coarse (9 cliques covering 87 tickers), and
  public knowledge already priced in.
- **Correlation edges** are derived from the same price series the model
  already consumes as node features. They re-encode existing information
  rather than adding any.
- **Co-mention edges** averaged 10.8 edges/day against sector's 381 — too
  sparse to carry signal, and measured as Experiment 6's worst relation
  (+0.0060 MCC).

Tweet-similarity edges are the first graph in this project derived from
information that is **neither price-derived nor static**.

## The macro / micro split

This distinction is architectural, not just terminology:

| Channel | Contents | Granularity |
|---|---|---|
| **Macro** | FNSPID headlines with **no** `Stock_symbol` (market-wide) + FOMC/CPI/NFP release calendar | One vector per trading day, shared by all tickers |
| **Micro** | Per-ticker tweets (`Company_Texts`, `Event_Texts`) + FNSPID headlines **tagged** with a universe ticker | One vector per (ticker, day) |

**Macro conditions micro.** The macro vector generates FiLM parameters that
transform each ticker's micro representation before it enters the graph.

Naming honesty: pooled market-wide financial news is **market-level news**,
not macroeconomics proper. The genuinely macroeconomic component is the
release calendar. The write-up names both accurately and never presents the
pooled-news channel as a macroeconomic indicator.

## The between-day / within-day problem

**This is the most important methodological constraint in the experiment,
and it was measured, not assumed.** Variance decomposition of the binary
target on the protocol-A splits:

| Split | Days | Between-day variance share | Days >80% one-directional | Oracle market-direction-only |
|---|---|---|---|---|
| train | 398 | 40.7% | 167/398 | acc 0.7811, MCC +0.5622 |
| dev | 42 | 56.8% | 24/42 | acc 0.8454, MCC +0.6863 |
| **test** | **64** | **48.3%** | **32/64** | **acc 0.8116, MCC +0.6248** |

On any given day the ~58 tickers in the cross-section mostly move together.
Roughly half the label variance is therefore *market direction*, not
*which stock outperforms which*. A predictor that knows only the day's
market direction and cannot discriminate between stocks at all scores
**MCC +0.6248** on the test set — far above every published number on this
benchmark (StockNet 0.081, MAN-SF 0.195).

Two consequences:

1. **Pooled MCC conflates two different abilities.** A model that partly
   learns market direction gains a lot of pooled MCC while discriminating
   between stocks not at all. The graph exists to do the second thing.
   Macro news conditioning is *precisely* the kind of signal that could
   improve the first, so Experiment 7 is the experiment most exposed to
   this confusion.
2. **Within-day performance must be reported.** For every row of the
   ablation ladder, alongside pooled metrics, report **within-day MCC**:
   compute MCC within each day's cross-section and average across days
   (skipping days with a single class present, and reporting how many were
   skipped). That number, not pooled MCC, is what says whether the graph
   discriminates stocks.

This is not leakage — market direction is legitimately part of the task as
StockNet defines it, and all published numbers are pooled. It is a
measurement-validity issue: the write-up must not present a pooled-MCC gain
driven by market-direction learning as evidence that the graph works.

## Scope

**In scope:** macro and micro news acquisition and caching; a
tweet-similarity edge builder; a macro-conditioned micro encoder; a GAT
over the learned graph; the ablation ladder that attributes any gain to the
right component; the explainability readout.

**Out of scope (deliberately):** the Wikidata company-relation graph;
cross-day recurrence over graph embeddings; federated learning. See
"Deferred".

## Global constraints

- Never describe results as reproducing or beating MAN-SF's GAT+price MCC
  of 0.117 or their headline 0.195. Those were measured on a Wikidata
  company-relation graph, which is not built here.
- Protocol A is unchanged. All runs use
  `dataset/stocknet_protocol_a_exact.parquet`; test `n_samples` must remain
  **3720**, matching Experiments 1/3/4/5/6.
- **5 seeds minimum**, not 3. Experiment 6 measured a 0.0247 noise floor and
  the effect being hunted is ~0.025. Report mean ± std always; a difference
  smaller than seed spread is "indistinguishable", never "an improvement".
  Every results table reports seed spread beside the mean.
- **Leakage is a tested property**, not an assumption. Every builder takes
  an explicit cutoff and has a test asserting that corrupting data at or
  after the target date cannot change that date's output.
- The MCC **> 0.15 tripwire** stays armed: any cell exceeding it is treated
  as suspected leakage to be re-audited, not a success.
- **English-only text.** All headlines are language-filtered before FinBERT;
  FinBERT is English-only and FNSPID contains non-English content (Russian
  observed in probing). The filter's reject rate is reported.
- Existing files are read-only except where a task says otherwise:
  `src/models/graph_model.py`, `src/models/graph_transformer.py`,
  `src/models/graph_models.py`, `src/data/graph_edges.py`, and all
  `results/experiment[1-6]*`.

---

## Task 0 — Fix the `Price3` feature (prerequisite)

Measured across all 106,476 day-slots in the protocol-A parquet:

| Feature | std | range |
|---|---|---|
| `Price1` | 0.065 | [−0.094, 0.489] |
| `Price2` | 0.063 | [−0.294, 0.395] |
| **`Price3`** | **154.0** | **[−6495, +6500]** |

1.33% of `Price3` values exceed \|10\|; 0.58% exceed \|100\|. After the global
z-score in `normalize_price_snapshots`, **99% of `Price3` compresses into
±0.103** while outliers reach \|z\| = 42. The channel is effectively constant
zero punctuated by spikes.

Values of ±6500 are impossible for a normalized price ratio, so this is a
construction defect upstream, not a market event. Every experiment using
this parquet — 3, 4, 5, 6 — has been running on two usable price features
and one degenerate one.

**Required:** diagnose the root cause; if it cannot be corrected at the
source, winsorize at a documented percentile. Then **re-run one Experiment 6
cell** (`GAT [sector] seed=0`, previously MCC +0.0263) to measure whether
the baseline itself moves. That number is reported in the write-up
regardless of direction — if the fix changes Experiment 6's conclusions,
that supersedes them.

This task gates everything else. A conditioning mechanism evaluated on
broken node features measures nothing.

---

## Component 1 — News acquisition (`src/data/fnspid.py`)

### Source

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
(`Stock_news/All_external.csv`, 5.34 GB, 15.7M records, 1999–2023,
CC BY-NC-4.0), from arXiv 2402.06698. Verified accessible without
authentication; the server supports range requests.

Schema, read from the live header:

```
Date, Article_title, Stock_symbol, Url, Publisher, Author,
Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary
```

Measured properties (20 byte-range probes, 524 rows parsed — **a clustered
sample, so these are estimates, not census figures**):

- `Date` carries **time-of-day in UTC** (`2014-04-23 14:31:02 UTC`).
- `Article_title` is reliably populated; `Article` and the four summary
  columns are usually empty.
- **~55%** of rows have an empty `Stock_symbol` — these become the macro
  channel. The remainder, where the symbol is in the 87-ticker universe,
  become the micro news channel.
- **~45%** of sampled rows fall in 2014–2015.
- Observed market-level headlines in-window: *"Market Wrap For April 23:
  S&P 500 Ends Winning Streak"*, *"Mid-Afternoon Market Update: Markets
  Trade in Choppy Session"*.

### Acquisition

The file is sorted by **neither date nor symbol** (probes at 10/30/50/70/90%
returned 2004, 2008, 2008, and mid-article text respectively), so there is
nothing to seek to — any filter reads the whole file.

Single streaming pass, raw file never stored:

```
stream 5.34 GB  →  filter: 2014-01-01..2015-12-31, English only
                →  split by Stock_symbol:
                     empty          -> macro headlines parquet
                     in universe    -> micro (per-ticker) headlines parquet
                →  data/processed/fnspid_{macro,micro}.parquet   (a few MB)
```

Measured throughput 2.7 MB/s → **~34 minutes**. I/O-bound, so it runs in
the background while model code is built and tested against synthetic
fixtures. It is not a blocking prerequisite.

Must be resumable and must not require the full file on disk at any point.

**Coverage gate:** the task reports per-trading-day headline counts for both
channels before any model work depends on them. **Macro-channel day coverage
below 70% of the 504 trading days triggers a redesign of the macro channel**
rather than silently proceeding.

### Per-day headline cap — required, not optional

The probe projects roughly **7.0M in-window rows**, of which ~55% are
symbol-less: on the order of **3.9M macro headlines, ~7,600 per trading
day**. FinBERT is BERT-base; at the 30–50 segments/sec obtainable from 16
CPU cores, embedding that corpus is **tens of hours** — it would outrun the
entire experiment.

Both channels are therefore capped per (day, channel) before any FinBERT
call: keep at most **N headlines per trading day**, selected by recency
relative to the cutoff, with `N = 64` as the starting default. The cap is
set for real against Component 1's coverage report, and the chosen value
plus the resulting total segment count are recorded in the write-up. Cap
selection must not consult labels.

### Macro calendar

A small hand-entered table of 2014–2015 **FOMC meeting dates, CPI release
dates, and non-farm-payroll release dates** — public, unambiguous, and
genuinely macroeconomic. Stored as `dataset/macro_calendar.csv`, checked
into the repo.

---

## Component 2 — Macro event vector (`src/data/macro_encoder.py`)

For trading day `t`:

```
m_t = concat(
    FinBERT_mean( macro headlines published strictly before t's market
                  open, within a 3-calendar-day lookback ),  # 768 -> 32 proj
    calendar_flags(t)                                        # FOMC/CPI/NFP
)                                                            # => m_t is 35-d
```

The 3-day lookback is a starting default, not a measured optimum; it is the
one macro hyperparameter, and Component 1's coverage report sets it against
real density rather than a guess.

**Leakage rule:** a headline counts toward day `t` only if its UTC timestamp
precedes `t`'s 09:30 ET market open. This is why the time-of-day field
matters, and it is unit-tested: corrupting all headlines at or after the
cutoff must not change `m_t`.

Days with no qualifying headlines yield a learned default vector, never a
crash and never a dropped sample.

---

## Component 3 — Micro signal (`src/data/micro_encoder.py`)

For each (ticker `i`, trading day `t`):

```
tweet_i,t = FinBERT_mean( ticker i's tweets in a 5-trading-day window
                          ending at t-1 )                    # 768-d
news_i,t  = FinBERT_mean( ticker i's FNSPID headlines
                          published before t's market open ) # 768-d

micro_i,t = proj( concat(tweet_i,t , news_i,t) )             # -> 32-d
```

**Interface requirement:** the encoder must expose `tweet_i,t`
**separately** from the pooled `micro_i,t`. Component 4 builds the graph
from the tweet half alone, so that graph structure and news content stay
separately attributable; if only the pooled vector is exposed, the graph
builder cannot honor that and the ablation ladder silently stops isolating
what it claims to.

Tweet text comes from `dataset/final/stocknet_final_modeling_set.parquet`
(`Company_Texts`, `Event_Texts`), joined on `(Ticker, Date)`. Corpus size,
measured: **64,097 company-tweet segments** across 12,571 rows (47.3%) and
**5,353 event-tweet segments** across 3,095 rows (11.6%). FinBERT over
~69,450 segments on CPU is a one-time cost of roughly 40–60 minutes, cached
to disk, plus the micro-news headlines from Component 1.

The **17 of 26,619** protocol-A samples uncovered by this join (carried
forward from Experiment 6) must degrade to an empty micro vector — never
crash, never be dropped. Tickers with neither tweets nor news in the window
receive a learned default vector.

---

## Component 4 — Tweet-similarity graph (`src/data/tweet_graph.py`)

For target date `t`:

1. For each ticker `i`, take its tweet-derived embedding over a window
   ending at **t−1** (Component 3's tweet half, before macro conditioning).
2. Edge `(i, j)` if `cos(v_i, v_j)` is among ticker `i`'s **top-k**
   (default `k = 5`).
3. Undirected, alphabetically-sorted tuples, no self-edges — the same
   contract as `src/data/graph_edges.py`, so `combine_edges` is reused
   unchanged.

**Top-k rather than a cosine threshold**, for the reason Experiment 6
documented: a threshold produced 57% density and rebuilt the over-smoothing
collapse diagnosed in Experiment 4. Top-k makes density a controlled
parameter.

The graph is built from **unconditioned** tweet embeddings, so that graph
structure and macro conditioning remain separately attributable in the
ablation ladder.

Tickers with no tweets in the window contribute no edges that day. Early
dates degrade to an empty graph rather than raising.

---

## Component 5 — `MacroConditionedGraphNet` (`src/models/macro_conditioned.py`)

```
              m_t (35-d macro vector, shared across all tickers on day t)
                              │
                              ▼
                    γ, β = MLP(m_t)              FiLM parameters
                              │
   micro_i,t (32-d) ──────────┤
                              ▼
              micro'_i,t = γ ⊙ micro_i,t + β     ← MACRO CONDITIONS MICRO
                              │
   price_i (4x3, Task-0 fixed)│
        │                     │
   PriceEncoder               │
        └──────── concat ─────┘
                    │
                    ▼  node features h_i
        GAT over the tweet-similarity graph (residual, 2 layers)
                    │
                    ▼
          per-ticker up/down logits (N, 2)
```

Design notes:

- FiLM was chosen over edge-gating (less expressive), regime-conditioning
  (coarse, arbitrary K), and a hypernetwork (unstable to train, unusable at
  a 0.0247 noise floor).
- **The conditioning point is the micro representation, not the attention
  logits.** The claim is semantic: an earnings headline means something
  different during a tightening cycle than during easing. Whether the
  attention logits are the better conditioning site is an empirical
  question, tested as row 6 of the ladder rather than assumed either way.
- Residual connections around each message-passing layer, matching
  `StockGAT` and `StockGraphSAGE`, for the over-smoothing reason documented
  in `results/experiment4_gat/EXPERIMENT4_GAT.md`.
- Must accept an empty `edge_index` and run.

---

## Component 6 — Ablation ladder

Every row is 5 seeds, and every row reports **pooled and within-day metrics
side by side**. This ladder, not any single number, is the experiment's
actual output. 8 rows × 5 seeds = 40 runs.

| # | Variant | Isolates |
|---|---|---|
| 0 | **macro-only: `m_t`, no micro, no graph** | **is this just market-direction prediction?** |
| 1 | price-only, no graph | Experiment 6 control, re-run post-Task-0 |
| 2 | + micro features (tweets+news), no graph | does per-ticker text help at all? |
| 3 | + tweet graph, no conditioning | does the learned graph help? |
| 4 | **+ macro→micro FiLM (full model)** | does macro conditioning help? |
| 5 | **shuffled-macro control** | **is it the events, or just parameters?** |
| 6 | macro→attention FiLM instead | is micro the right conditioning site? |
| 7 | sector graph + full conditioning | does conditioning help *any* graph? |

**Row 0 is mandatory**, and it is the cheapest row in the ladder. It has no
per-ticker input whatsoever, so it *cannot* discriminate between stocks —
every ticker on a given day receives an identical prediction. If row 0
captures most of row 4's pooled gain, then the finding is "macro news
predicts market direction," the graph contribution is nil regardless of
pooled MCC, and the write-up says so. Row 5 cannot detect this, because
shuffling the macro vector destroys both abilities at once.

Note the shared-`β_t` mechanism makes this concrete rather than
hypothetical: because `β_t` is identical across every ticker on day `t`, it
functions as a per-day intercept after the classifier, which is exactly a
market-direction control.

**Row 5 is mandatory and is the experiment's most important control.**
It re-runs row 4 with macro vectors randomly permuted across the calendar —
identical parameter count, identical architecture, destroyed
event-alignment. If row 5 matches row 4, the gain was capacity, not events,
and the headline claim is withdrawn. Row 5 runs at the same 5 seeds and is
reported beside row 4 in the results table, never omitted.

Row 6 tests the design decision in Component 5 rather than assuming it.
Row 7 guards the converse error: if conditioning improves the sector graph
just as much, the contribution is the conditioning mechanism, not the tweet
graph, and the write-up must say so.

---

## Component 7 — Explainability readout

The learned FiLM parameters are the interpretable object:

1. Plot `γ_t`, `β_t` across the 504 trading days against the macro calendar.
   Do FOMC days measurably shift how micro signal is transformed?
2. Rank days by \|γ_t − mean(γ)\| and report the top-10 macro headlines on
   those days. Does the model reweight micro signal on days a human would
   call eventful?
3. Compare the learned tweet graph's edges against sector and correlation
   edges: what fraction does it recover, what does it find that they miss?

**This readout is a deliverable independent of the MCC outcome.** If the
ablation ladder shows no MCC gain, "macro events measurably change how
per-ticker text is interpreted, and here is what changes" remains a
reportable finding. This is deliberate risk management: the capstone should
not depend on beating a benchmark.

Attention weights are **not** claimed as explanations of model reasoning —
that claim is contested (Jain & Wallace, *Attention is not Explanation*,
NAACL 2019). The claim made is narrower and defensible: the conditioning
signal measurably changes the micro representation, and the change
correlates with events.

---

## Evaluation and reporting

- Protocol A splits, `n_samples == 3720`, verified per run.
- Metrics, **pooled and within-day, reported side by side for every row**:
  accuracy, F1, MCC, AUC. Within-day MCC is computed per trading-day
  cross-section and averaged, reporting how many days were skipped for
  having a single class present. **Lead with AUC and within-day MCC** in the
  honest reading — Experiment 6 demonstrated that pooled MCC alone flatters
  cells sitting near a lucky threshold while their AUC is at or below
  chance, and the variance decomposition above shows pooled MCC additionally
  rewards market-direction learning that the graph is not responsible for.
- The market-direction oracle (**pooled MCC +0.6248, acc 0.8116**, zero
  stock discrimination) is reported in the results table as a reference
  row, so the reader can see what pooled MCC is capable of rewarding.
- Reference points carried into the results table: Experiment 1 (+0.0922),
  Experiment 6 best (+0.0677 ± 0.0255, RGCN sector+correlation),
  Experiment 6 GAT sector (+0.0160 ± 0.0161), always-UP constant
  (acc 0.5129 / F1 0.6780), always-DOWN (acc 0.4871 / F1 0.0).
- **Model selection must not consume the protocol dev set for tuning.**
  Selection folds are carved from *train*; the protocol-exact dev/test
  split is untouched for the reported number. Experiment 6's 42-day dev set
  is a plausible driver of its 0.0247 seed spread, and widening it would
  forfeit literature comparability — so the fix goes in the selection
  procedure, not the split boundaries.
- Results in `results/experiment7_macro_conditioned/`, `metrics.json`
  schema matching Experiments 1/3/4/5/6.
- Realistic target is MAN-SF's **GAT+price ablation at 0.117**, not their
  headline 0.195, which has known reproduction difficulty across the
  literature.

---

## Risks

1. **Three new components at once** (micro text features, learned graph,
   macro→micro conditioning). The ablation ladder is the mitigation —
   without it a positive result would be uninterpretable. The ladder is not
   optional scope that can be trimmed under time pressure.
2. **FNSPID 2014–15 yield is estimated from a clustered sample**, not
   measured. Component 1's coverage gate (70% of trading days) is the
   tripwire; below it, the macro channel is redesigned rather than patched.
3. **FiLM may simply not train** at this signal level. Row 3 (graph, no
   conditioning) is the fallback result and is worth reporting on its own.
4. **CPU-only.** Experiment 6's 84 cells took ~2 days wall-clock. This
   ladder is 35 runs, but each is heavier than an Experiment 6 cell because
   node features are wider. FinBERT caching adds a one-time ~40–60 minutes.
5. **Macro conditioning is shared across all tickers on a given day**, so it
   cannot by itself change the *relative* ranking of tickers within a day —
   only how their micro signal is scaled before the graph mixes them. Its
   effect therefore has to come through the interaction with the graph and
   with per-ticker micro magnitude. If rows 4 and 3 are indistinguishable,
   this is the first explanation to test.
6. **The pooled metric rewards the wrong ability** (see "The between-day /
   within-day problem"). Mitigated by mandatory row 0, within-day metrics on
   every row, and the oracle reference row — but it remains the failure mode
   most likely to produce a result that looks good and means little.
7. **FinBERT cost scales with the news channels, not the tweets.** The
   per-day cap in Component 1 is what keeps this bounded; without it the
   embedding job is tens of hours. If the cap has to be set very low to stay
   tractable, the macro channel gets correspondingly noisier, and that
   trade-off is reported rather than hidden.

## Deferred

**Cross-day recurrence (Experiment 8).** A GRU/LSTM over the sequence of
daily graph embeddings per ticker — currently every trading day is modeled
independently, with no memory across days. This is a genuinely promising and
distinct contribution, and is *not* what MAN-SF does (they encode price
temporally, then apply GAT once per day). Deferred so this experiment stacks
one architectural novelty rather than two. If Experiment 7 produces a
working conditioned graph, recurrence is the natural next step.

**Wikidata company-relation graph.** Cheap to obtain (Feng et al. published
relation encodings) and valuable as a control that anchors the results table
against a graph independently known to carry signal. Tracked separately so
it does not entangle with this experiment's novelty.
