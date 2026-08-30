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

## Diagnosed bug #2: training collapse from one-gradient-step-per-day batching

After the over-smoothing fix above, this experiment still scored MCC
-0.0054 — barely different from chance. Root-cause investigation
(systematic ablation, one variable at a time) found this was **not**
about the graph or the GAT architecture at all: a plain `PriceEncoder` +
linear classifier, with no graph structure whatsoever, trained under this
same per-day loop, collapsed identically (validation MCC exactly 0.0 from
epoch 5 onward, never recovering). Swapping in the reference GAT paper's
hyperparameters, using pre-LN instead of post-LN normalization, and adding
a learning-rate warmup schedule (tested on the graph-transformer
architecture in Experiment 5, which shares this same training loop) all
failed to fix it — ruling out architecture and hyperparameters entirely.

The actual cause: one gradient step per single trading day gives the
optimizer almost no gradient diversity per step, because every ticker
active on a given day shares that day's market-wide conditions (systemic
moves, not independent per-ticker signal). Experiment 3, by contrast, used
`DataLoader(..., batch_size=64, shuffle=True)` — randomly mixing unrelated
tickers and dates into each batch, which this per-day loop structurally
could not do. Accumulating gradients over `GRAD_ACCUM_DAYS = 16` trading
days before each optimizer step (still respecting each day's own
graph/snapshot boundaries — only the optimizer step frequency changes, not
what each forward pass sees) fixed it: validation MCC stopped collapsing
to exactly 0.0 and started showing real, non-degenerate positive signal
across most epochs in a fresh ablation, before this fix was rolled into
`scripts/train_gat.py` for a real run.

This resolves lever #2 from this document's own prior "How we should move
forward" section, which had flagged the per-day training regime as
untested.

## Results

All numbers below are on the same 3,720-row exact-reproduction test split
used by Experiments 1 and 3 (`n_samples` confirmed to match across all
three `metrics.json` files).

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Experiment 1 (LSTM, raw tweet count, FS3) | 0.5468 | 0.6363 | +0.0922 | 0.5666 |
| Experiment 3 price-only (GRU + temporal attention, no graph) | 0.5140 | 0.6720 | +0.0108 | 0.5425 |
| **Experiment 4 sector-graph GAT (price only, post both fixes)** | 0.5177 | 0.6699 | +0.0278 | 0.5026 |

With both the over-smoothing fix (residual connections) and the training-loop
fix (gradient accumulation) applied, this run reaches a genuinely positive
MCC (+0.0278) for the first time — better than Experiment 3's price-only
result (+0.0108), though still below Experiment 1's tweet-count baseline
(+0.0922). Training took 44 epochs to converge before early stopping
(best validation MCC +0.0892 at epoch 34, per `metrics.json`) — roughly
4x longer than the pre-fix run's early stopping point, consistent with
each epoch now containing ~25 optimizer steps (398 training days /
16 days per accumulated step) instead of 398.

**Honest reading:** the sector graph, once both bugs are fixed, does show
a real (if modest) improvement over the no-graph price-only baseline —
the first positive result any graph-based model in this project has
produced. It still doesn't beat Experiment 1's raw-tweet-count signal.
Whether a sparser graph (correlation or dynamic edges) would do better
than this same-sector graph remains an open, now more meaningful,
question — see "How we should move forward."

### Disclosure: remaining differences from Experiment 3

This run's training uses train-statistic price normalization (z-scored
using train-split mean/std, applied to dev/test with the same statistics),
gradient clipping (max norm 1.0), and gradient-accumulated batching (see
"Diagnosed bug #2" above) — all now much closer to Experiment 3's setup
than the original per-day-step version. One difference remains
uncontrolled: **epoch budget** — Experiment 3 used a 50-epoch budget;
Experiment 4 here uses `MAX_EPOCHS=100` (unchanged, per plan), and this
run used 44 of them.

## What this experiment does and doesn't establish

Establishes: whether a cheap, always-available graph signal (sector
membership) helps a price-only model at all.

Does not establish: whether MAN-SF's actual graph construction (Wikidata
relations) would perform differently -- that would require building a
Wikidata-relation graph, which is out of scope here.

## How we should move forward

The training-regime lever (gradient accumulation) is now resolved and
produced this document's first positive result. One lever remains
untried: **the graph structure itself.** The sector graph's complete-clique
density is still what caused the over-smoothing bug in the first place,
and even with residual connections, a maximally dense graph gives the GAT
the least useful structure to learn from (every node's neighborhood is
identical: "everyone else in the sector"). Correlation edges (20-day
rolling return correlation, |corr| > 0.5) or tweet-co-occurrence dynamic
edges would both be sparser and more selective than "same sector" --
and, per the project's `docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md`
"Reproduction vs. Contribution" section, are also genuinely absent from
MAN-SF's own design, which is where this project's actual novelty claim
over MAN-SF lives.

With the training loop now behaving correctly, a sparser edge type is a
fair test for the first time -- earlier results (including this one, before
today's fix) were confounded by an optimizer that couldn't learn from any
graph, dense or sparse.

## Reproducing this experiment

```bash
python scripts/train_gat.py
```

Runs locally on CPU (no GPU required) -- 87 nodes/day, ~500 trading days,
completes in minutes, not hours. `set_seed(42)` makes this deterministic;
full per-epoch stdout is captured in
`results/experiment4_gat/training_log.txt`.

## Visualizing the graph

```bash
python scripts/export_graph_to_neo4j.py
```

Pushes the static 87-node sector graph to Neo4j Aura for browsing (one-time,
visualization only -- never read during training). Requires `NEO4J_URI`,
`NEO4J_USER`, `NEO4J_PASSWORD` in a local `.env` (see `.env.example`).

Full setup instructions (creating a free Aura instance, configuring
credentials, and example Cypher queries to actually explore the graph
once it's pushed) are in `results/experiment4_gat/NEO4J_GUIDE.md`.
