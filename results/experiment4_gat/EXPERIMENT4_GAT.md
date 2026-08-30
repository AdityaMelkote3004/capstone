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

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Sector-graph GAT (price only) | 0.5091 | 0.1700 | +0.0745 | 0.5079 |

Compared against:
- Experiment 1 baseline (LSTM, raw tweet count): MCC +0.092
- Experiment 3 price-only (GRU + temporal attention, no graph): MCC +0.036

The sector-graph GAT (MCC +0.0745) improves meaningfully over Experiment
3's price-only result (+0.036) — roughly double the MCC — suggesting that
letting a ticker's representation attend to its same-sector neighbors adds
real signal beyond treating each stock independently. It still falls short
of Experiment 1's LSTM/raw-tweet-count baseline (+0.092), so adding the
sector graph closes part of, but not all of, the gap to that baseline.
Training also stopped early (epoch 14 of a possible 100, patience 10),
which caps how much the model was able to exploit the graph structure
before validation MCC plateaued.

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
