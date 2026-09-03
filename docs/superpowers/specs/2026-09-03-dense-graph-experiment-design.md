# Experiment 6 — Information-Dense Stock Graph + Model Grid — Design Spec

Date: 2026-09-03

## Purpose

Experiments 4 and 5 tested graph models on a **sector-only** graph and found
at best a marginal benefit (GAT +0.0278 MCC on a single seed, after two
bug fixes). Both experiments' own write-ups concluded the same thing: the
limiting factor is most likely the *graph itself* — "same sector" is a
coarse, static relationship — not the attention mechanism sitting on top
of it.

This experiment tests that conclusion directly. It builds two additional
edge types (rolling return correlation, tweet co-mention), runs every
non-empty combination of the three edge types across multiple graph model
architectures and seeds, and then adds tweet text on top of the winning
graph. It answers: **does a richer graph carry signal that a sector-only
graph does not?**

## Scope

In scope:
- `src/data/graph_edges.py` — three leakage-safe edge builders returning
  per-target-date edge sets, plus a combiner.
- `src/models/graph_models.py` — `StockGraphSAGE` and `StockRGCN`, plus a
  multi-relation extension of the existing graph-transformer bias.
- `notebooks/MMGTFFF_dense_graph_experiment.ipynb` — self-contained Colab
  notebook that clones the repo and runs the four passes below.
- `results/experiment6_dense_graph/` — `metrics.json` per run, a combined
  `results.csv`, `EXPERIMENT6_DENSE_GRAPH.md`, and PNG figures.
- Unit tests for every edge builder, with explicit leakage assertions.

Out of scope (deferred):
- Neo4j export of the new edge types. Deferred by the user's own
  sequencing ("we can do neo4j once everything fully works"); the edge
  builders are designed so the existing
  `scripts/export_graph_to_neo4j.py` can consume them later without
  rework.
- Federated training (the project's eventual Phase 5).
- Fundamentals as a modality (price + text + graph only here).

## Datasets and the cross-dataset join

Two files are involved, and this is deliberate:

- `dataset/stocknet_protocol_a_exact.parquet` (26,619 samples) is the
  **source of truth for samples, labels, price windows, and the
  train/dev/test split** — unchanged from Experiments 1/3/4/5, so results
  are directly comparable (test `n_samples` must remain 3720).
- `dataset/final/stocknet_final_modeling_set.parquet` (26,603 rows)
  supplies **`Return`** (for correlation edges) and **`Event_Texts` /
  `Company_Texts`** (for co-mention edges and the text encoder), joined
  on `(Ticker, Date)`.

Join coverage verified before this spec was written: **26,602 of 26,619
exact-protocol samples are covered** (17 missing, 0.06%), same 87 tickers,
same 2014-01-02→2015-12-31 range, zero nulls in `Return`. The 17
uncovered samples must be handled explicitly (they get an empty
correlation/co-mention neighbourhood, never a crash), not silently
dropped.

## Edge builders (`src/data/graph_edges.py`)

All three return, for a given target date `t`, a list of undirected
`(ticker_a, ticker_b)` pairs. All take an explicit cutoff so leakage is a
tested property, not an assumption.

### 1. Sector edges — `build_sector_edges(sector_csv)`
Static. Two tickers are connected iff they share a `Sector`. 87 nodes,
381 edges, 9 disjoint complete cliques. Identical to Experiment 4's graph
(reuses `src/models/graph_model.py::build_sector_graph`), so the
sector-only cells of this grid are a direct re-run of Experiment 4 and
serve as the multi-seed sanity check on its `+0.0278` result.

### 2. Correlation edges — `build_correlation_edges(returns, target_date, window=60, k=5, min_periods=10)`
For target date `t`: take the **60 trading days strictly before `t`**
(i.e. ending at `t-1`), compute the pairwise Pearson correlation of daily
returns, and connect each ticker to its **top-`k` most correlated
partners by |ρ|**.

**Top-k, not a global threshold.** Measured during design: a `|ρ| > 0.5`
threshold yields **2,130 edges — 5.6x denser than the sector graph's
381**, covering 57% of all possible pairs. Graph density is precisely what
caused Experiment 4's over-smoothing collapse, so a threshold-based
correlation graph would reproduce that failure mode by construction.
Top-k gives direct density control, guarantees no isolated nodes, and
makes density a tunable axis rather than an emergent accident. Measured at
`window=60, k=5`: **~330 edges** steady-state, comparable to the sector
graph's 381 and 8.8% density. `k` is a parameter; `k=5` is the default.

**Window is 60 days, not 20, because of the sparsity of the return
matrix.** The matrix is **39% NaN**, and this is *not* mid-period
listings: **all 87 tickers have gaps inside their own active range**, and
no ticker exceeds 404 of 504 trading days (worst: AGFS at 145). Each
ticker is missing roughly 40% of days, scattered.

The practical consequence, measured across the full date range:

| Window | Ticker pairs with >=10 overlapping days | Avg valid partners per ticker |
|---|---|---|
| 20 days | 28.7% of 3741 | 24.9 of 86 |
| 40 days | 93.2% | 75.0 |
| **60 days** | **97.7%** | **82.9** |
| 90 days | 97.7% | 84.2 |

At a 20-day window, top-k selection would be drawn from a candidate set
restricted to whichever tickers happened to trade on the same days — the
resulting "most correlated partners" would encode **data availability, not
market behaviour**, a confound invisible in the results table. 60 days is
the point where coverage saturates (90 days buys nothing further), so it
is the default.

**`min_periods=10`**: pairs with fewer than 10 overlapping return
observations in the window produce no edge. Without this, pandas computes
a correlation of 1.0 from two shared observations. At `window=60` this
guard now rejects only ~2.3% of pairs rather than the 71% it would reject
at `window=20`.

**Warm-up behaviour (documented, not a bug)**: the first trading days of
the dataset have insufficient history for a full window. Measured:
day 5 yields 0 correlation edges, day 15 yields 63, and the graph reaches
its steady ~330 by roughly day 40. This affects only ~30 of the 398
training days (7.5%); the dev and test splits begin far later and are
entirely unaffected. Days with insufficient history return an **empty
correlation neighbourhood**, never an error — nodes fall back to whatever
other edge types the combination includes, or to self-loops alone.

**Leakage guard**: the window is `[t-window, t-1]` — it never includes `t`
or anything after. Unit-tested by constructing a returns series where the
target day's return is an extreme outlier and asserting the resulting
edges are identical with and without that day present.

### 3. Co-mention edges — `build_comention_edges(event_texts_by_date, target_date)`
Two tickers are connected on day `t` if they appear together in the same
tweet on the day(s) feeding `t`. Cashtags are extracted with
`\$\s*([A-Za-z]{1,5})` (the corpus format is `$ aapl`, with a space) and
intersected with the 87-ticker universe; a tweet mentioning 2+ universe
tickers contributes an edge between each pair.

Measured during design: **1,785 tweet segments mention 2+ universe
tickers, spread across 347 of 504 trading days** — this is a live signal,
not a near-empty relation as an early read of the 12%-of-rows figure
suggested.

**Leakage guard**: tweets are attached to trading days using Experiment
1's existing convention — a calendar day's tweets attach to the **first
trading day strictly after** that calendar day, never the same day.
Tweets from day `t` itself never inform the prediction for day `t`.
Unit-tested.

### Combining edge types
`combine_edges(edge_sets)` returns both:
- a **merged** `edge_index` (union, deduplicated, both directions) for
  GAT / GraphTransformer / GraphSAGE, and
- an **`edge_type`** vector aligned to it (0=sector, 1=correlation,
  2=co-mention) for R-GCN, which is the one model that treats them as
  genuinely distinct relations.

Where two edge types both connect the same pair, the merged view keeps one
edge; the typed view assigns it the lowest-numbered relation, and this
tie-break is documented in the function's docstring.

## Models (`src/models/graph_models.py`)

All take the same node features (per-ticker `PriceEncoder` output,
optionally concatenated with a text embedding) and produce per-ticker
up/down logits. All reuse `PriceEncoder` from
`src/models/price_encoder.py` unchanged.

- **`StockGAT`** — existing, unchanged, imported from
  `src/models/graph_model.py`. Residual-wrapped 2-layer `GATConv`.
- **`StockGraphTransformer`** — existing, extended: its single learned
  scalar bias becomes **one learned scalar per edge type**, so the model
  can weight "same sector" differently from "correlated" and
  "co-mentioned". With only sector edges present this reduces exactly to
  Experiment 5's model.
- **`StockGraphSAGE`** — new. 2-layer `SAGEConv`, residual-wrapped for
  consistency with `StockGAT`. Samples/aggregates neighbours rather than
  attending over them; included because neighbour aggregation degrades
  more gracefully on dense graphs than attention did in Experiment 4.
- **`StockRGCN`** — new. 2-layer `RGCNConv` with `num_relations=3`,
  residual-wrapped. The only model here that learns a separate
  transformation per edge type.

`GATConv`, `TransformerConv`, `SAGEConv`, and `RGCNConv` were all verified
importable and functional in this environment before this spec was
written.

## Text encoder (Pass 4)

Reuses the `HierarchicalTweetEncoder` architecture from Experiments 2/3
(tweet-level attention → daily GRU → temporal attention), embedding-
agnostic, run with two backends: **FinBERT** (`ProsusAI/finbert`, 768-d
CLS) and **USE** (512-d). Per-tweet embeddings are computed once and
cached to disk as `.npy` before any training begins.

Fusion: the 64-d text representation is **concatenated to each ticker's
node features before the graph layers**, so text and graph structure
interact inside the message passing rather than being merged after it.
This is the specific weakness Experiment 2 diagnosed in its own late-
concatenation design.

## The four passes

Each pass appends to the same results table; later passes never re-run
completed cells.

| Pass | Grid | Runs |
|---|---|---|
| 1 | 7 edge combos x {GAT, GraphTransformer} x 1 seed | 14 |
| 2 | extends Pass 1 to 3 seeds | 42 cumulative |
| 3 | adds {GraphSAGE, R-GCN} at 3 seeds | 84 cumulative |
| 4 | winning graph x {FinBERT, USE} text x 3 seeds | 90 cumulative |

"7 edge combos" = all non-empty subsets of {sector, correlation,
co-mention}.

**"Winning graph" is defined operationally**, not by judgement: the
`(model, edge_combo)` cell with the highest **mean test MCC across its 3
seeds** after Pass 3. If two cells are within one standard deviation of
each other, the one with the **fewer edge types** wins (prefer the
simpler graph), and the tie-break is recorded in the write-up.

For `StockRGCN`, `num_relations` is always 3 even when the edge combo
contains fewer relations — the unused relation's weights simply receive
no gradient. This keeps one model definition across all 7 combos rather
than reconstructing the model per combo.

## Crash resilience

This project has lost Colab runs twice (an OOM crash, and a background
process killed mid-run), so this is a hard requirement, not a nicety:

- Every completed run appends one row to
  `results/experiment6_dense_graph/results.csv` and writes its own
  `metrics.json` **immediately on completion**, before the next run
  starts.
- On startup the notebook reads the existing results and **skips any
  `(model, edge_combo, seed, text_backend)` cell already present**.
- Per-day correlation and co-mention edge sets are precomputed once and
  cached to disk (`.pkl`) before training; tweet embeddings are cached to
  `.npy`. Both are reloaded on restart rather than recomputed.

## Figures

Written to `results/experiment6_dense_graph/figures/` as PNG:
- `graph_sector.png`, `graph_correlation.png`, `graph_comention.png` —
  networkx renders, nodes coloured by sector, showing the structural
  difference between the three relations.
- `mcc_by_model_and_edges.png` — grouped bar chart of test MCC across the
  grid, with Experiment 1 (+0.0922), 3 (+0.0108), and 4 (+0.0278) drawn
  as reference lines.
- `edge_density_over_time.png` — edge counts per trading day per edge
  type, showing how dynamic the correlation and co-mention graphs are.
- `seed_variance.png` — per-cell MCC spread across seeds, so single-seed
  noise is visible rather than implied.

## Evaluation and reporting

`EXPERIMENT6_DENSE_GRAPH.md` follows the Experiment 4/5 format: results
table against all prior baselines, an honest statement of whether the
richer graph beats the sector-only graph, the standard MAN-SF
non-reproduction caveat, a disclosure section for uncontrolled
differences, and next steps.

**Reporting requirements specific to this experiment:**
- Report **mean ± std across seeds**, never a single seed's number. A
  difference smaller than the seed spread must be described as
  indistinguishable, not as an improvement.
- Include the **always-UP / always-DOWN constant-predictor baselines**
  (accuracy 0.5129 / F1 0.6780 and accuracy 0.4871 / F1 0.0 respectively
  on this test split), which currently beat every model this project has
  produced on accuracy and F1. Any claim of success is measured against
  those, not only against prior experiments.
- **Leakage tripwire**: if any cell produces MCC substantially above the
  ~0.09 range this project has established (say >0.15), the write-up must
  treat leakage as the primary hypothesis and report the result as
  *pending verification* until the edge builders' cutoffs are re-audited
  for that configuration.

## Risks

- **Density recreating over-smoothing.** Mitigated by top-k correlation
  edges and residual connections in every model, but a 3-edge-type union
  is still denser than sector alone. The grid measures this directly —
  the sector-only cells are the control.
- **Compute.** 84 price-only runs on a Colab GPU is comfortable; Pass 4's
  text embedding of ~106k tweets through FinBERT is the expensive step
  (~10-20 min on GPU), which is why it is cached and runs last.
- **The 17 uncovered samples** from the cross-dataset join must degrade
  to empty neighbourhoods rather than crashing or being dropped.
