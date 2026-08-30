# Graph Transformer (Global Attention) — Design Spec

Date: 2026-08-30

## Purpose

Experiment 4's sector-graph GAT was diagnosed with a root-cause bug:
message passing restricted to graph neighbors (`GATConv`) over a
complete-clique sector graph over-smooths within 1-2 layers, collapsing
every ticker's representation to a near-identical constant. GAT only
attends to a ticker's direct graph neighbors — never to tickers outside
its sector, no matter how relevant. This experiment (Experiment 5)
replaces local, neighbor-restricted attention with **global attention**:
every ticker attends to every other ticker in that day's cross-section,
with graph structure (currently: same-sector membership) injected as an
**additive bias to the raw attention scores**, matching the mechanism used
by Graphormer-style graph transformers. This sidesteps the over-smoothing
mechanism (there is no repeated local-neighborhood-averaging step to
compound) while still letting the model learn to weight same-sector
relationships differently from unrelated ones.

## Scope

In scope:
- `src/models/graph_transformer.py`:
  - `build_sector_bias(tickers: list, graph: nx.Graph) -> torch.Tensor`
    (shape `(N, N)`): a dense structural-bias matrix for one day's ticker
    list, entry `[i, j]` = a single learned scalar parameter if tickers
    `i` and `j` share a sector, else `0.0`. Diagonal (self) entries are
    `0.0` (self-attention is already handled natively by
    `nn.TransformerEncoderLayer`, no separate self-loop bias needed).
  - `StockGraphTransformer` (`nn.Module`): `PriceEncoder` (reused
    unmodified from `src/models/price_encoder.py`) → 2 ×
    `nn.TransformerEncoderLayer` (the bias matrix passed as each layer's
    `src_mask`, which `nn.MultiheadAttention` adds to raw attention scores
    before softmax when given as a float tensor, per PyTorch's documented
    behavior) → linear classifier head (`hidden_dim -> 32 -> 2`, matching
    `StockGAT`'s head exactly for a fair comparison).
- `scripts/train_graph_transformer.py`: mirrors `scripts/train_gat.py`
  structurally (same data loading via `build_daily_snapshots` +
  `normalize_price_snapshots`, same optimizer settings — Adam lr=1e-3,
  weight_decay=1e-4, grad clipping norm 1.0 — same early-stopping
  patience=10/max epochs=100, same `sys.path` shim), swapping only the
  model and the bias-construction call (`build_sector_bias` instead of
  `edge_index_for_tickers`).
- `results/experiment5_graph_transformer/metrics.json` +
  `EXPERIMENT5_GRAPH_TRANSFORMER.md`, same schema/convention as
  Experiments 1-4, explicitly comparing against Experiment 4's sector-GAT
  result (MCC -0.0054) and Experiment 3's price-only baseline (MCC
  +0.0108).
- Tests: `build_sector_bias` (correct scalar placement for same-sector
  pairs, zero for different-sector pairs, zero diagonal, symmetric);
  `StockGraphTransformer` forward-shape test; a collapse-regression test
  mirroring Experiment 4's (`test_stock_gat_does_not_collapse_...`)
  adapted for this model, since global attention can still over-smooth
  with enough layers/training, just less severely than local
  message-passing over a complete graph.

Out of scope (explicitly deferred):
- Multiple relation types / continuous edge weights (correlation
  strength, tweet co-occurrence) — the bias matrix is sector-only for
  now, per this session's decision to keep it minimal rather than
  generalize ahead of having those edge types built. Revisit when
  correlation/dynamic edges exist.
- Any change to `StockGAT`/Experiment 4's own files — this is an
  additive new experiment, not a replacement. Experiment 4's results and
  code stay exactly as they are for side-by-side comparison.
- Text/fundamentals branches — price-only, matching Experiment 4's scope
  exactly for a controlled comparison (only the graph mechanism changes).

## Architecture

```
Daily cross-section of tickers with a valid sample that day
        |
Per-ticker: PriceEncoder (GRU + temporal attention, unchanged)
        |
Structural bias matrix for this day: (N, N) float tensor,
  bias[i,j] = learned scalar if tickers i, j share a sector, else 0.0
        |
2 x nn.TransformerEncoderLayer (multi-head self-attention; the bias
  matrix is added to raw attention scores before softmax via each
  layer's src_mask argument; every ticker attends to every other
  ticker in the day's cross-section, not just same-sector ones;
  includes the standard feed-forward + residual + layer-norm block)
        |
Per-ticker classifier head (hidden_dim -> 32 -> 2)
        |
UP / DOWN
```

**Why this addresses the diagnosed bug:** Experiment 4's failure came from
stacking two *local*, neighbor-restricted aggregation steps over a
complete graph, where "average of your neighbors" converges to nearly the
same value for every node in a fully-connected group. Global attention has
no such restriction — every ticker can attend to every other ticker
directly in one layer, and the structural bias only *reweights* that
existing global attention rather than defining the sole channel through
which information can flow. Two layers of global attention do not compound
the "everyone's neighborhood is the same set" pathology the way two layers
of clique-restricted message passing did.

**Bias implementation detail:** `nn.TransformerEncoderLayer`'s `src_mask`
parameter, when passed as a `torch.float` tensor (not `bool`), is added
directly to attention scores before the softmax
(`softmax(QK^T/sqrt(d) + mask)`) — this is exactly the Graphormer
mechanism, using PyTorch's own built-in support rather than a hand-rolled
attention implementation. The single learned scalar (`nn.Parameter`,
initialized near 0) controls how much extra attention weight same-sector
pairs receive relative to the baseline (unbiased) global attention; it is
learned jointly with the rest of the model via backprop, not hand-tuned.

## Data Flow

Identical to Experiment 4 up through snapshot construction:
```
Experiment 1's exact-reproduction samples (26,619 ticker-date rows)
        |
Group by main_target_date -> one daily snapshot per trading day
  (via build_daily_snapshots, unchanged, reused from Experiment 4)
        |
normalize_price_snapshots (train-stat z-scoring, unchanged, reused)
        |
Per-ticker: PriceEncoder -> 64-d
        |
build_sector_bias(tickers, graph) -> (N, N) bias matrix (NEW, replaces
  edge_index_for_tickers -- no edge_index/edge list at all in this model)
        |
2 x nn.TransformerEncoderLayer(d_model=64, nhead=4, ...) with bias as
  src_mask
        |
Per-ticker classifier head -> up/down logits
```

## Model Details

- `d_model=64` (matches `StockGAT`'s `hidden_dim`), `nhead=4` (matches
  `StockGAT`'s `heads`), `dim_feedforward=128` (standard 2x d_model,
  reasonable default not previously used elsewhere in this project since
  no prior experiment used a Transformer block — chosen as a conventional
  starting point, not tuned), `dropout=0.2` (matches `StockGAT`'s
  default), `batch_first=True`.
- `build_sector_bias` returns a matrix of a single scalar value
  (`self.sector_bias` as an `nn.Parameter`, shared across both
  Transformer layers and all attention heads) placed at same-sector
  off-diagonal positions. This is intentionally the simplest possible
  version of "structural bias" — one learned number, not a per-relation
  embedding vector — consistent with the decision to keep this
  sector-only and minimal rather than build unused generality.
- No `edge_index`, no `torch_geometric` message-passing layers in this
  model at all — `torch_geometric` remains a dependency (still used by
  Experiment 4 and `build_sector_graph`, which this experiment reuses
  unmodified for sector lookup), but `StockGraphTransformer` itself is
  built entirely from `torch.nn` primitives.

## Training Loop

Identical to `scripts/train_gat.py`'s: one gradient step per trading-day
snapshot, shuffled day order per training epoch, dev/test order
unshuffled, Adam (lr=1e-3, weight_decay=1e-4), gradient clipping (max norm
1.0), early stopping on validation MCC (patience=10, max 100 epochs),
`n_samples`/`best_epoch`/`best_val_mcc` persisted to `metrics.json`
alongside accuracy/f1/mcc/auc — same schema/conventions as Experiment 4,
so the two experiments are comparable line-for-line.

## Evaluation / Reporting

`results/experiment5_graph_transformer/EXPERIMENT5_GRAPH_TRANSFORMER.md`
reports Accuracy/F1/MCC/AUC on the test split, in a comparison table
against Experiment 1 (+0.0922), Experiment 3 price-only (+0.0108), and
Experiment 4 sector-GAT (-0.0054) explicitly. Must state plainly whether
global attention improves on, matches, or still fails to beat these
baselines — no rounding a marginal or negative result up, matching this
project's established honesty convention (see Experiment 3/4's writeups).
Must also carry the same MAN-SF non-reproduction disclaimer already
established for Experiment 4 (this remains a sector-based proxy graph,
not MAN-SF's Wikidata-relation graph, regardless of which attention
mechanism sits on top of it).

## Dependencies / Risks

- No new dependencies — `torch.nn.TransformerEncoderLayer` is part of
  core PyTorch, already installed.
- Risk: dense `(N, N)` attention with `N` up to ~90 tickers/day is
  trivially cheap (~8,100 entries), no memory/performance concern
  expected on CPU.
- Risk (flagged, not blocking): global attention can still over-smooth
  with enough layers or training time, just via a different, generally
  slower mechanism than clique message-passing. The collapse-regression
  test (see Scope) exists specifically to catch this if it happens, the
  same way Experiment 4's own regression test caught its bug.
