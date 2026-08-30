# Experiment 5 — Graph Transformer (Global Attention, Price Only)

## Objective

Test whether replacing GAT's local, neighbor-restricted attention with
global attention (every ticker attends to every other ticker in a day's
cross-section) plus a learned same-sector attention bias fixes or improves
on Experiment 4's result, given that Experiment 4's failure was diagnosed
as an over-smoothing bug specific to local message passing over a
complete-clique graph.

## Important: same non-reproduction caveat as Experiment 4

This remains a sector-based proxy graph, not MAN-SF's Wikidata-relation
graph -- see `docs/superpowers/specs/2026-08-30-graph-transformer-design.md`
and `results/experiment4_gat/EXPERIMENT4_GAT.md` for the full framing.
Nothing here should be read as reproducing or beating MAN-SF's GAT+price
figure (MCC 0.117).

## Architecture

```
Daily cross-section of tickers with a valid sample that day
        |
Per-ticker: PriceEncoder (GRU + temporal attention, unchanged)
        |
Structural bias: (N,N) matrix, learned scalar at same-sector pairs, 0
  elsewhere -- added to raw attention scores before softmax
        |
2 x nn.TransformerEncoderLayer (global self-attention: every ticker
  attends to every other ticker, not just same-sector ones)
        |
Per-ticker classifier head
        |
UP / DOWN
```

## Results

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Experiment 1 (LSTM, raw tweet count, FS3) | 0.5468 | 0.6363 | +0.0922 | 0.5666 |
| Experiment 3 price-only (GRU + temporal attention, no graph) | 0.5140 | 0.6720 | +0.0108 | 0.5425 |
| Experiment 4 sector-graph GAT (price only, post over-smoothing fix) | 0.5099 | 0.6618 | -0.0054 | 0.4953 |
| **Experiment 5 graph transformer (price only)** | 0.4871 | 0.0021 | -0.0008 | 0.5249 |

Experiment 5 does not improve on Experiment 4 in any way that matters, and
in one respect it is worse. The MCC (-0.0008) is nominally closer to zero
than Experiment 4's (-0.0054), but both are statistically indistinguishable
from a coin flip -- neither result carries real predictive signal. More
tellingly, the F1 score collapsed to 0.0021, versus 0.6618 for Experiment 4
and 0.6720 for Experiment 3: the trained model almost never predicts the
positive class, meaning it converged to (near-)always predicting "down"
rather than learning any decision boundary. Training itself corroborates
this -- `best_epoch` was 0, i.e. validation MCC never improved past its
very first-epoch value before the patience counter expired at epoch 10, and
train MCC oscillated around zero for the entire run rather than trending
upward. Global attention did not close the gap to Experiment 3's
price-only, no-graph baseline (+0.0108 MCC), let alone Experiment 1's
tweet-count baseline (+0.0922 MCC); if anything, replacing GAT's local
attention with global attention made the model harder to train usefully on
this graph, not easier. Fixing Experiment 4's over-smoothing bug clearly
did not translate into global attention finding exploitable structure in a
same-sector graph -- these were separate hypotheses, and this run answers
the second one negatively.

## What this experiment does and doesn't establish

Establishes: whether global attention (with the over-smoothing mechanism
architecturally avoided) can find useful signal in same-sector structure
that local GAT message-passing could not, once GAT's own bug is fixed.
The answer here is no: with the over-smoothing mechanism sidestepped by
construction, the sector-graph transformer still failed to learn a useful
decision boundary (MCC ~0, degenerate near-constant predictions), which
points at the sector-only graph itself, not the local-vs-global attention
mechanism, as the binding constraint.

Does not establish: whether a sparser or different graph (correlation
edges, dynamic tweet-co-occurrence edges, or MAN-SF's own Wikidata
relations) would perform differently under either attention mechanism --
the edge definition (same-sector) is held constant across Experiments 4
and 5 specifically so the comparison isolates the attention mechanism,
not the graph.

## How we should move forward

Experiment 5 does not improve on Experiment 4, and both remain well below
Experiments 1 and 3. Since fixing the attention mechanism (local -> global,
with over-smoothing architecturally ruled out) did not help, the evidence
now points at the sector-only graph itself, not the choice of attention
mechanism, as the limiting factor. The next step should be building a
richer edge type -- 20-day rolling return correlation edges or dynamic
tweet-co-occurrence edges -- per `results/experiment4_gat/RELATED_WORK.md`,
rather than further attention-mechanism tuning on the same sector graph.
Price-only signal without a graph (Experiment 3) and tweet-count signal
(Experiment 1) both still outperform either graph variant tried so far,
so any future graph-based model should be judged against those two
baselines, not against Experiment 4 or 5.

## Reproducing this experiment

```bash
python scripts/train_graph_transformer.py
```

Runs locally on CPU (no GPU required) -- dense attention over at most ~90
tickers/day is inexpensive, completes in minutes.
