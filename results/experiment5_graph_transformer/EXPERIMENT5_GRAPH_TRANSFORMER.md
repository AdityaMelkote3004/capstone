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
this graph, not easier.

### Constant-predictor comparison

The test split (n=3720, positive/"up" rate 0.5129) makes it possible to
check Experiment 5 against the two trivial constant classifiers directly:

| Predictor | Accuracy | F1 | MCC |
|---|---:|---:|---:|
| Always predict DOWN (constant) | 0.4871 | 0.0000 | 0.0000 |
| Always predict UP (constant) | 0.5129 | 0.6780 | 0.0000 |
| Experiment 5 (actual model) | 0.4871 | 0.0021 | -0.0008 |

Experiment 5's test accuracy (0.4870967741935484) is bit-for-bit identical
to the always-predict-DOWN constant classifier's accuracy on this split.
Back-solving the confusion matrix from accuracy/F1/MCC gives the unique
integer solution TP=2, FP=2, TN=1810, FN=1906: the model made only ~4
positive ("UP") predictions total across 3720 test examples (2 correct, 2
incorrect) -- this is not "near-collapse," it *is* the always-DOWN
constant predictor plus negligible noise. That is much stronger, directly
verifiable evidence that
the model degenerated to a near-constant predictor than the F1-vs-Experiment-4
comparison above.

Worth noting separately: the always-UP constant predictor alone (F1 0.6780,
MCC 0.0000) beats Experiments 3, 4, *and* 5 on both accuracy and F1 (though
of course not on MCC, where a constant predictor is definitionally zero).
That is useful context for how weak all three graph/price-only results are
in absolute terms -- the comparisons in this document are relative to each
other and to Experiment 1, not to a trivial baseline that any of them could
be checked against.

## What this experiment does and doesn't establish

Establishes: whether this training run, as configured, produces a useful
sector-graph transformer. It does not. `best_epoch` was 0, but that does
not mean the model barely trained: training takes one gradient step per
trading-day snapshot, there are 398 unique training dates, so epoch 0
alone is 398 optimizer steps, and the full run (11 epochs, 0 through 10,
before early stopping at patience=10) is ~4,378 optimizer steps in total
(confirmed by re-running and capturing `results/experiment5_graph_transformer/training_log.txt`).
The optimizer ran plenty of steps -- it just never made progress: train
loss only ever reached 0.6943 (essentially chance-level cross-entropy,
ln(2)≈0.6931), and validation MCC never exceeded its epoch-0 value
(0.0216) across all 11 epochs, oscillating between roughly -0.015 and
+0.022 with no trend. That is an optimization/architecture failure: the
model ran for thousands of steps without ever reaching a useful trained
state, as distinct from reaching one and finding no signal there. The AUC
(0.5249) is actually above Experiment 4's (0.4953), i.e. some ranking
signal exists in the model's raw scores even though the thresholded
predictions collapsed to a single class -- more consistent with a
collapsed decision head / stuck optimization than with "zero signal
anywhere in the graph."

This run does *not* establish that the sector-only graph itself carries no
signal for global attention to exploit. That would require a run whose
loss actually moved off chance level and still failed to generalize --
this one didn't get that far (loss bottomed out at chance-level ln(2),
despite thousands of optimizer steps). Whether the sector-only graph, the
transformer architecture, or the optimization setup (learning rate, the
single scalar sector-bias parameterization, lack of a warmup/schedule) is
the binding constraint remains genuinely open and is not settled by this
run.

Does not establish: whether a sparser or different graph (correlation
edges, dynamic tweet-co-occurrence edges, or MAN-SF's own Wikidata
relations) would perform differently under either attention mechanism --
the edge definition (same-sector) is held constant across Experiments 4
and 5 specifically so the comparison holds the graph constant. This does
*not* mean the comparison isolates the attention mechanism alone -- see
"Disclosure" below for the other architectural differences between the
two models.

### Disclosure: architectural differences beyond attention mechanism

Holding the same-sector graph constant across Experiments 4 and 5 controls
one variable, but the two model blocks differ in more than local-vs-global
attention:
- **LayerNorm and feedforward sublayers**: `StockGAT` is a single residual
  block, `h = h + relu(GATConv(h))`, with no LayerNorm anywhere.
  `StockGraphTransformer` is two full post-LN `nn.TransformerEncoderLayer`
  blocks -- each with two LayerNorms and a 128-unit feedforward sublayer,
  plus attention dropout. That is at least 3-4 architectural differences
  (normalization placement, presence of a feedforward sublayer, dropout,
  depth of nonlinearity per layer) beyond which tickers attend to which.
- **No learning-rate warmup**: post-LN transformers (as used here) are
  well documented in the literature as training-unstable without a
  learning-rate warmup schedule; this run uses a flat `lr=1e-3` for all
  epochs. That instability is a plausible independent contributor to this
  run's failure to leave chance-level loss, separate from whatever the
  same-sector graph does or doesn't offer global attention.
- **Epoch budget vs Experiment 3**: matching Experiment 4's own disclosed
  gap, Experiment 3 used a 50-epoch budget; Experiment 5 here uses
  `MAX_EPOCHS=100` (same setting as Experiment 4), so the epoch budget is
  matched to Experiment 4 but not to Experiment 3.

None of this changes the run's bottom line (it failed to learn), but it
does mean "global vs local attention" is not the only thing that changed
between Experiments 4 and 5, and any future attempt to isolate the
attention mechanism specifically would need to equalize these too.

## How we should move forward

Experiment 5 does not improve on Experiment 4, and both remain well below
Experiments 1 and 3. But because this run shows an optimization/architecture
failure (best_epoch 0, thousands of optimizer steps but loss stuck at
chance level) rather than a converged model that found no signal, it would
be premature to conclude the sector-only graph is the limiting factor from
this run alone. Two things are worth trying before drawing that
conclusion: (1) re-run this exact architecture with a lower learning rate,
a warmup schedule, and/or more capacity on the single-scalar sector-bias
term, to see if it can actually reach a trained state; if it still lands
at MCC ~0 with train loss that has actually moved off chance level, that would
be much stronger evidence against the sector-only graph. (2) In parallel,
build a richer edge type -- 20-day rolling return correlation edges or
dynamic tweet-co-occurrence edges -- per
`results/experiment4_gat/RELATED_WORK.md`, since that question (does a
different graph help) is independent of whether this particular training
run converges. Price-only signal without a graph (Experiment 3) and
tweet-count signal (Experiment 1) both still outperform either graph
variant tried so far, so any future graph-based model should be judged
against those two baselines, not against Experiment 4 or 5.

## Reproducing this experiment

```bash
python scripts/train_graph_transformer.py
```

Runs locally on CPU (no GPU required) -- dense attention over at most ~90
tickers/day is inexpensive, completes in minutes. `set_seed(42)` is called
at the start of `main()`, so this is deterministic: a re-run for this fix
pass reproduced the committed `metrics.json` and per-epoch numbers cited
above bit-for-bit. Full per-epoch stdout from that re-run is captured in
`results/experiment5_graph_transformer/training_log.txt`.
