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
2-layer GATConv (4 heads) over that day's same-sector edges
        |
Per-ticker classifier head
        |
UP / DOWN
```

## Results

All numbers below are on the same 3,720-row exact-reproduction test split
used by Experiments 1 and 3 (`n_samples` confirmed to match across all
three `metrics.json` files).

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Experiment 1 (LSTM, raw tweet count, FS3) | 0.5468 | 0.6363 | +0.0922 | 0.5666 |
| Experiment 3 price-only (GRU + temporal attention, no graph) | 0.5140 | 0.6720 | +0.0108 | 0.5425 |
| **Experiment 4 sector-graph GAT (price only)** | 0.4871 | 0.0000 | 0.0000 | 0.4628 |

(Experiment 3's real price-only number is MCC +0.0108, not the +0.036
previously and incorrectly cited in this document — that figure does not
appear anywhere in this repo's results. Source:
`results/experiment3_mansf_bilinear/final_run/Price_Only/metrics.json`.)

This run (post price-normalization fix, see disclosure below) trained for
11 epochs before early stopping (patience 10), with the best validation
MCC recorded at **epoch 0** (best_val_mcc = 0.0). In other words, model
selection never found an epoch past initialization-level training where
the model did anything other than collapse to predicting a single class:
F1 = 0.0 means the model never predicts the positive class on the test
set at all, and AUC = 0.4628 is *below* 0.5 — the model's ranking of
examples by predicted probability is anti-informative, not merely
uninformative. Accuracy (0.4871) sits below what a majority-class
classifier would achieve if the test set's positive rate exceeds ~48.7%.

None of this supports "adds real signal." On every one of the four
metrics, this run is worse than Experiment 3's price-only baseline (which
itself is only barely above chance), and it is also worse than the
previous, incorrect version of this experiment's own number
(accuracy 0.5091, F1 0.1700, MCC +0.0745, AUC 0.5079) — which was itself
already a near-degenerate classifier being trained on unnormalized inputs
with an outlier price feature reaching |values| > 6500. The honest reading
is that the previously reported "improvement" was most likely an artifact
of how the unnormalized data happened to interact with GRU gating and
training dynamics in that specific (uncontrolled) run, not evidence that
sector-neighbor attention adds predictive signal. With price inputs
properly z-scored and the optimizer brought in line with Experiment 3,
this sector-graph GAT does not demonstrate any benefit over a price-only,
non-graph baseline on this dataset/split, and in this run it fails to
learn a non-degenerate classifier at all.

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

Per the project roadmap, the next graph extensions are correlation edges
(20-day rolling return correlation, |corr| > 0.5) and tweet-co-occurrence
dynamic edges -- both of which, unlike this sector graph, are genuinely
absent from MAN-SF's own design (see
`docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md`'s
"Reproduction vs. Contribution" section) and are where this project's
actual novelty claim over MAN-SF lives.

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
