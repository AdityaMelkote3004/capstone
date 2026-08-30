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

## Diagnosed bug: training collapse from one-gradient-step-per-day batching

The results below supersede an earlier run (MCC -0.0008, F1 0.0021,
best_epoch 0) whose "Establishes" section originally called this run's own
failure "genuinely open" as to cause. It is no longer open: root-cause
investigation (systematic ablation, one variable at a time, prompted by
the same underlying question as this document's earlier "How we should
move forward" section) found the failure had nothing to do with the graph,
the attention mechanism, LayerNorm placement, or hyperparameters. A plain
`PriceEncoder` + linear classifier, with **no graph or attention layer at
all**, trained under this experiment's same one-gradient-step-per-trading-day
loop, collapsed identically (validation MCC exactly 0.0 from epoch 5
onward). Pre-LN normalization, a learning-rate warmup schedule, and a
lower learning rate (all tested here, per this document's own prior
recommendation) each failed to fix it too.

The actual cause: one gradient step per single trading day gives the
optimizer almost no gradient diversity per step, since every ticker active
on a given day shares that day's market-wide conditions. Experiment 3, by
contrast, used `DataLoader(..., batch_size=64, shuffle=True)` -- randomly
mixing unrelated tickers and dates into each batch. Accumulating gradients
over `GRAD_ACCUM_DAYS = 16` trading days before each optimizer step (still
respecting each day's own graph/attention-mask boundaries -- only the
optimizer step frequency changes) fixed the collapse in a fresh ablation
before being rolled into `scripts/train_graph_transformer.py` for the real
run below. The same fix was applied to Experiment 4's `train_gat.py`,
where it produced that experiment's first genuinely positive result
(+0.0278 MCC) -- see `results/experiment4_gat/EXPERIMENT4_GAT.md`.

## Results

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| Experiment 1 (LSTM, raw tweet count, FS3) | 0.5468 | 0.6363 | +0.0922 | 0.5666 |
| Experiment 3 price-only (GRU + temporal attention, no graph) | 0.5140 | 0.6720 | +0.0108 | 0.5425 |
| Experiment 4 sector-graph GAT (price only, post both fixes) | 0.5177 | 0.6699 | +0.0278 | 0.5026 |
| **Experiment 5 graph transformer (price only, post training-loop fix)** | 0.4871 | 0.0185 | -0.0026 | 0.5458 |

The literal collapse is gone: the model made a real spread of predictions
during training (best validation MCC +0.0764 at epoch 5, `metrics.json`),
not a bit-identical constant one. But the fix did not turn this into a
working model the way it did for Experiment 4. Validation MCC oscillated
between roughly -0.11 and +0.076 across 16 epochs with no clear upward
trend (`results/experiment5_graph_transformer/training_log.txt`), and the
checkpoint selected by early stopping (epoch 5's validation peak) did not
generalize: test MCC is -0.0026, and F1 (0.0185) is still close to the
always-predict-DOWN constant classifier, though no longer bit-identical to
it (see "Constant-predictor comparison" below).

**Honest reading:** this is a different, more ordinary failure than
before -- an overfitting/instability pattern (validation signal appears,
doesn't hold up on test) rather than the earlier literal training
collapse. Experiment 4's simpler residual-GAT block reached a real,
generalizing positive result under the identical training-loop fix;
Experiment 5's larger post-LN Transformer block (two LayerNorms and a
feedforward sublayer per layer, no learning-rate warmup) did not, which is
consistent with the architectural differences already disclosed below
mattering independently of local-vs-global attention.

### Constant-predictor comparison

The test split (n=3720, positive/"up" rate 0.5129) makes it possible to
check Experiment 5 against the two trivial constant classifiers directly:

| Predictor | Accuracy | F1 | MCC |
|---|---:|---:|---:|
| Always predict DOWN (constant) | 0.4871 | 0.0000 | 0.0000 |
| Always predict UP (constant) | 0.5129 | 0.6780 | 0.0000 |
| Experiment 5 (actual model, post-fix) | 0.4871 | 0.0185 | -0.0026 |

Test accuracy is still bit-for-bit identical to the always-DOWN constant
classifier (0.4870967741935484), but F1 (0.0185, up from 0.0021 pre-fix)
shows the model now makes noticeably more than ~4 positive predictions
total -- it is closer to, but no longer indistinguishable from, a pure
constant predictor.

Worth noting separately: the always-UP constant predictor alone (F1 0.6780,
MCC 0.0000) still beats Experiments 3, 4, *and* 5 on both accuracy and F1
(though not on MCC, where a constant predictor is definitionally zero) --
useful context for how weak all of these price-only results remain in
absolute terms.

## What this experiment does and doesn't establish

Establishes: the earlier run's failure was a training-loop bug (fixed),
not evidence about the sector-only graph or global attention specifically
-- confirmed by the fact that a graph-free model failed identically under
the old loop. With that bug fixed, this experiment now establishes that
*this particular* Transformer block (post-LN, no warmup, two LayerNorms +
feedforward per layer) still does not generalize on this graph, even
though it did produce real (non-constant) predictions and some validation
signal.

Does not establish: whether a sparser or different graph (correlation
edges, dynamic tweet-co-occurrence edges, or MAN-SF's own Wikidata
relations) would perform differently under either attention mechanism --
the edge definition (same-sector) is held constant across Experiments 4
and 5 specifically so the comparison holds the graph constant. This does
*not* mean the comparison isolates the attention mechanism alone -- see
"Disclosure" below for the other architectural differences between the
two models, which now look like the more likely explanation for Exp4 vs
Exp5's diverging results under the identical, now-fixed training loop.

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

This is now the more likely explanation for Experiment 5's continued
underperformance relative to Experiment 4, since the training-loop bug
that previously confounded the comparison is fixed in both.

## How we should move forward

With the training-loop bug fixed, Experiment 4's simpler residual-GAT
block produced this project's first genuinely positive graph result
(+0.0278 MCC) while this Transformer block did not (-0.0026 MCC) --
under an identical graph and an identical (now-correct) training loop.
That points at the architectural differences disclosed above (LayerNorm
placement, feedforward sublayer, no learning-rate warmup on a post-LN
block) as the more likely explanation now, rather than local-vs-global
attention itself. Two concrete next steps, in priority order:

1. **Add a learning-rate warmup schedule and/or switch to pre-LN
   (`norm_first=True`)** for this Transformer block specifically -- both
   are standard, well-documented fixes for exactly the post-LN
   instability pattern disclosed above, and neither requires touching the
   graph or the global-attention mechanism this experiment exists to test.
2. **Build a richer edge type** -- 20-day rolling return correlation edges
   or dynamic tweet-co-occurrence edges, per
   `results/experiment4_gat/RELATED_WORK.md` -- now that Experiment 4 has
   shown the sector graph *can* carry real signal once trained correctly,
   this question is more meaningful than it was before today's fix.

Experiment 4's price-only baseline (+0.0278) is now this project's
best price-only graph result and the bar any future graph-based model
(including a re-tuned version of this Transformer) should be judged
against, alongside Experiment 3 (+0.0108, no graph) and Experiment 1
(+0.0922, tweet count).

## Reproducing this experiment

```bash
python scripts/train_graph_transformer.py
```

Runs locally on CPU (no GPU required) -- dense attention over at most ~90
tickers/day is inexpensive, completes in minutes. `set_seed(42)` is called
at the start of `main()`, so this is deterministic: a re-run reproduced
the committed `metrics.json` and per-epoch numbers cited above bit-for-bit.
Full per-epoch stdout is captured in
`results/experiment5_graph_transformer/training_log.txt`.
