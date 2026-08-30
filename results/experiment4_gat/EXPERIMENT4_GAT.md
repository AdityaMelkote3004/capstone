# Experiment 4 — Sector-Graph GAT (Price Only)

## Objective

Test whether letting each ticker's price representation attend to its
same-sector neighbors (a 2-layer GAT) improves on treating every ticker as
independent, as Experiments 1 and 3 did.

## Important: this is not a reproduction of MAN-SF's GAT+price figure

MAN-SF's own GAT+price ablation result (MCC 0.117) was measured on a graph
built from Wikidata company relations (first/second-order relations like
"owned by," "board member of," shared founder/industry) -- verified against
the primary source (`aclanthology.org/2020.emnlp-main.676`, Table 2 and
Section 4.4). This experiment uses a materially simpler graph: an edge
between two tickers exists purely if they share the same `Sector` label
(9 sectors, 87 tickers). This is a cheaper proxy, not the same experiment,
and its result must be read as its own number, not a reproduction or
refutation of MAN-SF's 0.117.

## Architecture

```
Daily cross-section of tickers with a valid sample that day
        |
Per-ticker: PriceEncoder (GRU + temporal attention, from Experiment 3)
        |
2-layer GATConv (4 heads) over that day's same-sector edges,
each layer wrapped in a residual connection: h = h + GATConv(h)
        |
Per-ticker classifier head
        |
UP / DOWN
```

The residual connections were added after diagnosing a training collapse —
see "Diagnosed bug: over-smoothing on a complete-clique graph" below.
Without them, the GAT layers erase each ticker's own signal entirely.

## Diagnosed bug: over-smoothing on a complete-clique graph

An earlier version of this experiment (no residual connections) trained to
a completely degenerate model: validation MCC stuck at exactly 0.0 every
epoch, test F1 = 0.0 (never predicts the positive class), test AUC = 0.4628
(below chance). Root-cause investigation (controlled ablation, changing one
variable at a time) ruled out hyperparameters as the cause: swapping in the
original GAT paper's reference settings (lr=0.005, weight_decay=5e-4,
dropout=0.6) produced the identical collapse. Removing the GAT layers
entirely (price encoder + classifier only, same training loop) did *not*
collapse — predictions stayed input-dependent.

Direct inspection of intermediate activations on real validation data found
the mechanism: before any GAT layer, per-ticker representations were
genuinely distinct (std 0.0106 across tickers). After a single `GATConv`
layer, every ticker's output vector was identical to 6 decimal places,
despite attention weights that were not themselves degenerate (spread
0.11–0.50). The sector graph is 9 disjoint **complete cliques** — every
ticker connected to every other ticker in its sector — and message passing
over a complete graph over-smooths rapidly, because "attention-weighted
average of your neighbors" is nearly the same quantity for every node when
your neighbors are "everyone else in the group." Two stacked layers fully
erased each ticker's distinguishing signal into the sector's shared mean.

Fix: wrap each `GATConv` layer in a residual connection so the classifier
always retains each ticker's own representation alongside the
sector-averaged one. A regression test
(`tests/test_graph_model.py::test_stock_gat_does_not_collapse_nodes_on_a_complete_clique_after_training`)
trains briefly on a synthetic complete-clique graph and asserts node
outputs don't collapse to a shared constant; it fails on the pre-fix code
and passes after the fix.

## Results

All numbers below are on the same 3,720-row exact-reproduction test split
used by Experiments 1 and 3 (`n_samples` confirmed to match across all
three `metrics.json` files).

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Experiment 1 (LSTM, raw tweet count, FS3) | 0.5468 | 0.6363 | +0.0922 | 0.5666 |
| Experiment 3 price-only (GRU + temporal attention, no graph) | 0.5140 | 0.6720 | +0.0108 | 0.5425 |
| **Experiment 4 sector-graph GAT (price only, post-fix)** | 0.5099 | 0.6618 | -0.0054 | 0.4953 |

The collapse is fixed: F1 (0.6618) and AUC (0.4953) show the model is
making varied, input-dependent predictions again, not a constant output.
But the result itself is still not a win — test MCC is slightly *negative*
(-0.0054), essentially indistinguishable from chance, and AUC sits just
below 0.5. Best validation MCC during training was +0.0556 at epoch 3
(`best_epoch`/`best_val_mcc` in `metrics.json`), meaning some signal was
found during training but it did not generalize to the test split.

**Honest reading:** fixing the over-smoothing bug turned a completely
broken model into a working-but-uninformative one. This sector-graph GAT
does not currently demonstrate a benefit over Experiment 3's price-only
baseline (+0.0108) or Experiment 1's tweet-count baseline (+0.0922) on
this dataset/split. Whether that's because sector membership genuinely
carries little price-prediction signal, or because the training regime
(one gradient step per day, on graphs that are maximally dense complete
cliques) still needs further tuning, is not yet resolved — see "How we
should move forward."

### Disclosure: remaining differences from Experiment 3

This run's training now uses train-statistic price normalization
(z-scored using train-split mean/std, applied to dev/test with the same
statistics) and gradient clipping (max norm 1.0), matching Experiment 3.
Two differences remain uncontrolled and are worth keeping in mind when
interpreting the gap to Experiment 3:
- **Epoch budget**: Experiment 3 used a 50-epoch budget; Experiment 4
  here still uses `MAX_EPOCHS=100` (unchanged, per plan).
- **Batching strategy**: Experiment 4 takes one gradient step per
  trading-day snapshot (all tickers active that day, in whatever order
  the snapshot list is shuffled to that epoch); Experiment 3 uses
  shuffled `batch_size=64` sampling over individual training examples.
  These are structurally different optimization regimes and are not
  equalized by this fix pass.

## What this experiment does and doesn't establish

Establishes: whether a cheap, always-available graph signal (sector
membership) helps a price-only model at all.

Does not establish: whether MAN-SF's actual graph construction (Wikidata
relations) would perform differently -- that would require building a
Wikidata-relation graph, which is out of scope here.

## How we should move forward

Two independent levers remain untried, and either could plausibly move
this result:

1. **The graph structure itself.** The sector graph's complete-clique
   density is what caused the over-smoothing bug in the first place, and
   even with residual connections, a maximally dense graph gives the GAT
   the least useful structure to learn from (every node's neighborhood is
   identical: "everyone else in the sector"). Correlation edges (20-day
   rolling return correlation, |corr| > 0.5) or tweet-co-occurrence dynamic
   edges would both be sparser and more selective than "same sector" --
   and, per the project's `docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md`
   "Reproduction vs. Contribution" section, are also genuinely absent from
   MAN-SF's own design, which is where this project's actual novelty claim
   over MAN-SF lives.
2. **The training regime.** One gradient step per trading-day snapshot is
   a structurally different (and likely noisier, less iid) optimization
   regime than Experiment 3's shuffled `batch_size=64` sampling over
   individual examples -- disclosed as an uncontrolled difference below.
   Whether batching multiple days together per step (e.g. via
   `torch_geometric.data.Batch`) changes this result is untested.

Recommend trying a sparser edge type (correlation or dynamic) before
concluding graph structure doesn't help here at all -- the complete-clique
sector graph is close to a worst case for GNN message passing, not a
representative one.

## Reproducing this experiment

```bash
python scripts/train_gat.py
```

Runs locally on CPU (no GPU required) -- 87 nodes/day, ~500 trading days,
completes in minutes, not hours.

## Visualizing the graph

```bash
python scripts/export_graph_to_neo4j.py
```

Pushes the static 87-node sector graph to Neo4j Aura for browsing (one-time,
visualization only -- never read during training). Requires `NEO4J_URI`,
`NEO4J_USER`, `NEO4J_PASSWORD` in a local `.env` (see `.env.example`).
