# Experiment 7 — Event-Conditioned Tweet Graph — Design Spec

**Status:** design approved 2026-09-08, not yet implemented.
**Predecessor:** `2026-09-03-dense-graph-experiment-design.md` (Experiment 6).

## Purpose

Experiment 6 measured every combination of hand-designed edges (sector,
rolling correlation, tweet co-mention) across four graph models and three
seeds, and found **no distinguishable improvement** over a sector-only
graph: mean within-cell seed spread was 0.0247, larger than the gap between
the first- and third-ranked cells, and 8 of 28 (model, edge) cells sat at
AUC below 0.51.

Experiment 7 changes two things Experiment 6 held fixed, and nothing else:

1. **The graph is learned from tweet semantics** rather than hand-designed
   from sector membership or price correlation.
2. **The graph's attention is conditioned on macro events** — a daily
   news-derived vector modulates attention day by day, so the effective
   graph is not static across the test period.

The question it answers: *does a graph built from what people say, and
reshaped by what is happening in the market, carry signal that sector
cliques and price correlation do not?*

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

## Scope

**In scope:** macro news acquisition and caching; a tweet-similarity edge
builder; a FiLM-conditioned GAT; the ablation ladder that attributes any
gain to the right component; the explainability readout.

**Out of scope (deliberately):** the Wikidata company-relation graph (a
separate control, tracked independently); cross-day recurrence over graph
embeddings (deferred — see "Deferred"); federated learning; per-ticker news
features from FNSPID (acquired but unused in this experiment).

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
- **Leakage is a tested property**, not an assumption. Every builder takes
  an explicit cutoff and has a test asserting that corrupting data at or
  after the target date cannot change that date's output.
- The MCC **> 0.15 tripwire** stays armed: any cell exceeding it is treated
  as suspected leakage to be re-audited, not a success.
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

## Component 1 — Macro event data (`src/data/macro_news.py`)

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
- **~55%** of rows have an empty `Stock_symbol` — general market news.
- **~45%** of sampled rows fall in 2014–2015.
- Observed market-level headlines in-window: *"Market Wrap For April 23:
  S&P 500 Ends Winning Streak"*, *"Mid-Afternoon Market Update: Markets
  Trade in Choppy Session"*.

### Naming honesty

Pooled financial news is **market-level news**, not macroeconomics. The
write-up must call it that. Genuine macroeconomic events enter through a
separate channel (below), and the two are never conflated.

### Acquisition

The file is sorted by **neither date nor symbol** (probes at 10/30/50/70/90%
returned 2004, 2008, 2008, and mid-article text respectively), so there is
nothing to seek to — any filter reads the whole file.

Single streaming pass, raw file never stored:

```
stream 5.34 GB  →  filter to 2014-01-01..2015-12-31,
                   keep rows with empty Stock_symbol (market-wide)
                   plus rows matching the 87-ticker universe
                →  data/processed/fnspid_headlines.parquet  (a few MB)
```

Measured throughput 2.7 MB/s → **~34 minutes**. I/O-bound, so it runs in
the background while model code is built and tested against synthetic
fixtures. It is not a blocking prerequisite.

Must be resumable and must not require the full file on disk at any point.

### Macro calendar

A small hand-entered table of 2014–2015 **FOMC meeting dates, CPI release
dates, and non-farm-payroll release dates** — public, unambiguous, and
genuinely macroeconomic. Stored as `dataset/macro_calendar.csv`, checked
into the repo (it is a few hundred rows).

### Daily macro vector

For trading day `t`:

```
m_t = concat(
    FinBERT_mean( headlines published strictly before t's market open,
                  within a lookback of 3 calendar days ),  # 768 -> 32 learned proj
    calendar_flags(t)                                      # FOMC/CPI/NFP, 3 binary
)                                                          # => m_t is 35-d
```

The 3-day lookback is a starting default, not a measured optimum; it is the
one macro hyperparameter, and the acquisition task reports per-day headline
counts so it can be set against real density rather than guessed.

**Leakage rule:** a headline counts toward day `t` only if its UTC
timestamp precedes `t`'s 09:30 ET market open. This is why the time-of-day
field matters, and it is unit-tested: corrupting all headlines at or after
the cutoff must not change `m_t`.

Days with no qualifying headlines yield a learned default vector, never a
crash and never a dropped sample.

---

## Component 2 — Tweet-similarity graph (`src/data/tweet_graph.py`)

Tweet text comes from `dataset/final/stocknet_final_modeling_set.parquet`
(`Company_Texts`, `Event_Texts`), joined on `(Ticker, Date)`. Corpus size,
measured: **64,097 company-tweet segments** across 12,571 rows (47.3%) and
**5,353 event-tweet segments** across 3,095 rows (11.6%). FinBERT over
~69,450 segments on CPU is a one-time cost of roughly 40–60 minutes,
cached to disk.

The **17 of 26,619** protocol-A samples uncovered by this join (carried
forward from Experiment 6) must degrade to empty neighbourhoods — never
crash, never be dropped.

### Edge construction

For target date `t`:

1. For each ticker `i`, mean-pool FinBERT embeddings of its tweets in a
   window ending at **t−1** (default lookback 5 trading days) → `v_i`.
2. Edge `(i, j)` if `cos(v_i, v_j)` is among ticker `i`'s **top-k**
   (default `k = 5`).
3. Undirected, alphabetically-sorted tuples, no self-edges — the same
   contract as `src/data/graph_edges.py`, so `combine_edges` is reused
   unchanged.

**Top-k rather than a cosine threshold**, for the reason Experiment 6
documented: a threshold produced 57% density and rebuilt the over-smoothing
collapse diagnosed in Experiment 4. Top-k makes density a controlled
parameter.

Tickers with no tweets in the window contribute no edges that day. Early
dates degrade to an empty graph rather than raising.

---

## Component 3 — `EventConditionedGAT` (`src/models/event_conditioned_gat.py`)

```
price (4x3, Task-0 corrected) ──PriceEncoder──┐
                                              ├─→ node features h_i
ticker tweet embedding (FinBERT → 32-d) ──────┘
                                    │
                                    ▼
        GAT over the tweet graph, attention logits FiLM-modulated:
            γ, β = MLP(m_t)                    # per attention head
            attn'_ij = γ ⊙ attn_ij + β
                                    │
                                    ▼
                        per-ticker up/down logits (N, 2)
```

Design notes:

- FiLM was chosen over edge-gating (less expressive), regime-conditioning
  (coarse, arbitrary K), and a hypernetwork (unstable to train, unusable at
  a 0.0247 noise floor).
- Residual connections around each message-passing layer, matching
  `StockGAT` and `StockGraphSAGE`, for the over-smoothing reason documented
  in `results/experiment4_gat/EXPERIMENT4_GAT.md`.
- This is also the first model in the project whose **node features carry
  text**. Experiments 4, 5, and 6 were all price-only — a fact that
  materially weakens their negative findings about graphs and is stated as
  such in the write-up.
- Must accept an empty `edge_index` and run.

---

## Component 4 — Ablation ladder

Every row is 5 seeds. This ladder, not any single number, is the
experiment's actual output.

| # | Variant | Isolates |
|---|---|---|
| 1 | price-only, no graph | Experiment 6 control, re-run post-Task-0 |
| 2 | + tweet node features, no graph | does text on nodes help? |
| 3 | + tweet graph, no conditioning | does the learned graph help? |
| 4 | + FiLM conditioning (**full model**) | does event-conditioning help? |
| 5 | **shuffled-news control** | **is it the news, or just parameters?** |
| 6 | sector graph + FiLM | does conditioning help *any* graph? |

**Row 5 is mandatory and is the experiment's most important control.**
It re-runs row 4 with news dates randomly permuted across the calendar —
identical parameter count, identical architecture, destroyed
event-alignment. If row 5 matches row 4, the gain was capacity, not events,
and the headline claim is withdrawn. Row 5 is run at the same 5 seeds and
reported beside row 4 in the results table, never omitted.

Row 6 guards the converse error: if FiLM improves the sector graph just as
much, the contribution is the conditioning mechanism, not the tweet graph,
and the write-up must say so.

---

## Component 5 — Explainability readout

The learned FiLM parameters are the interpretable object:

1. Plot `γ_t`, `β_t` across the 504 trading days against the macro calendar.
   Do FOMC days measurably shift attention modulation?
2. Rank days by \|γ_t − mean(γ)\| and report the top-10 headlines on those
   days. Does the model amplify graph attention on days a human would call
   eventful?
3. Compare the learned tweet graph's edges against sector and correlation
   edges: what fraction does it recover, what does it find that they miss?

**This readout is a deliverable independent of the MCC outcome.** If the
ablation ladder shows no MCC gain, "attention over a tweet-derived graph
restructures measurably around macro events, and here is what it attends
to" remains a reportable finding. This is deliberate risk management: the
capstone should not depend on beating a benchmark.

Attention weights are **not** claimed as explanations of model reasoning —
that claim is contested (Jain & Wallace, *Attention is not Explanation*,
NAACL 2019). The claim made is narrower and defensible: the conditioning
signal measurably changes attention, and the change correlates with events.

---

## Evaluation and reporting

- Protocol A splits, `n_samples == 3720`, verified per run.
- Metrics: accuracy, F1, MCC, AUC. **Lead with AUC** in the honest reading —
  Experiment 6 demonstrated that MCC alone flatters cells sitting near a
  lucky threshold while their AUC is at or below chance.
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
- Results in `results/experiment7_event_conditioned/`, `metrics.json`
  schema matching Experiments 1/3/4/5/6.
- Realistic target is MAN-SF's **GAT+price ablation at 0.117**, not their
  headline 0.195, which has known reproduction difficulty across the
  literature.

---

## Risks

1. **Three new components at once** (text node features, learned graph,
   FiLM conditioning). The ablation ladder is the mitigation — without it a
   positive result would be uninterpretable. The ladder is not optional
   scope that can be trimmed under time pressure.
2. **FNSPID 2014–15 yield is estimated from a clustered sample**, not
   measured. If the real yield is far lower, the macro channel thins out.
   The acquisition task must report actual per-day headline counts before
   any model work depends on them, and a day-coverage figure below ~70% of
   trading days triggers a redesign of the macro channel rather than
   silently proceeding.
3. **FiLM may simply not train** at this signal level. Row 3 (graph, no
   conditioning) is the fallback result and is worth reporting on its own.
4. **CPU-only.** Experiment 6's 84 cells took ~2 days wall-clock. This
   ladder is 6 rows × 5 seeds = 30 runs, comparable to a third of that, but
   the FinBERT caching adds a one-time ~40–60 minutes.
5. **Non-English headlines** are present in FNSPID (Russian observed in a
   probe). FinBERT is English-only; the filter should either language-gate
   or document the contamination.

## Deferred

**Cross-day recurrence.** A GRU/LSTM over the sequence of daily graph
embeddings per ticker — currently every trading day is modeled
independently, with no memory across days. This is a genuinely promising
and distinct contribution (and is *not* what MAN-SF does, which encodes
price temporally then applies GAT once per day). It is deferred so that
this experiment stacks one architectural novelty, not two. If Experiment 7
produces a working conditioned graph, recurrence is the natural
Experiment 8.

**Wikidata company-relation graph.** Cheap to obtain (Feng et al. published
relation encodings) and valuable as a control that anchors the results
table against a graph independently known to carry signal. Tracked
separately so it does not entangle with this experiment's novelty.
