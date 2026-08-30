# Sector-Graph GAT + Neo4j Visualization — Design Spec

Date: 2026-08-30

## Purpose

Build Experiment 4: a Graph Attention Network (GAT) that lets each ticker's
price representation be updated by attending to its same-sector neighbors,
tested against Experiment 1's baseline (LSTM, MCC +0.092) and Experiment 3's
price-only result (GRU+temporal attention, MCC +0.036). Also stand up a
Neo4j visualization of the sector graph so the structure can be inspected
directly, independent of training.

This is explicitly step 1 of a longer roadmap (see "Reproduction vs.
Contribution" below), not a claim of reproducing MAN-SF's GAT+price ablation
row (MCC 0.117) — that number was measured on a Wikidata company-relation
graph (verified against the primary source, `aclanthology.org/2020.emnlp-main.676`),
not a sector graph. Sector edges are a cheaper, different, and explicitly
labeled proxy.

## Reproduction vs. Contribution (read this before citing results)

**What is reproduction (attributed, not original):**
- Experiment 1's data pipeline ports `DataPipe.py`/`config.yml` logic from
  `yumoxu/stocknet-code` (MIT license, citation required — Xu & Cohen 2018).
  Attribution must be explicit in the notebook header and the paper's
  methodology section.
- Experiments 2/3's `PriceEncoder` / `HierarchicalTweetEncoder` /
  `BilinearFusion` are reimplementations of Sawhney et al. (EMNLP 2020,
  MAN-SF) built from the paper's architecture description. Cite as
  "we reproduce/adapt Sawhney et al. (2020)'s architecture" — never as
  original invention.
- This experiment's `GATConv` usage over a sector graph is loosely inspired
  by MAN-SF's GAT+price component, but is a materially different, simpler
  graph (same-sector membership vs. Wikidata first/second-order company
  relations). Report whatever MCC this produces on its own terms. Do not
  describe it as reproducing or beating the 0.117 figure.

**What is this project's actual contribution (not present in MAN-SF at all,
verified against the paper text):**
- Fundamentals as a third modality (MAN-SF uses price + text only).
- Correlation edges and tweet-co-occurrence dynamic edges (MAN-SF's graph
  is static and Wikidata-relation-based only — no price correlation, no
  tweet-driven edges, no temporal edge updates).
- Sector-based federated training across 9 clients (MAN-SF is fully
  centralized; federation is absent from it entirely).
- (Future, not this spec) Event-conditioned graph attention, where tweet
  content modulates edge attention rather than a fixed edge set.

This section exists so the eventual paper draws this line explicitly rather
than retrofitting it after results exist.

## Scope

In scope:
- `src/models/graph_model.py`: sector graph construction (NetworkX) over the
  88 StockNet tickers, using the `Sector` column already present in the
  parquet dataset; a `StockGAT` model (per-ticker `PriceEncoder` reused from
  Experiment 3 + 2-layer `GATConv`, 4 heads, torch_geometric).
- `scripts/export_graph_to_neo4j.py`: one-time push of the static graph
  (88 `Ticker` nodes with `sector`/`name` properties, `SAME_SECTOR`
  relationships) to a local Neo4j instance via the `neo4j` Python driver.
  Visualization only — never read at training time.
- `scripts/train_gat.py`: local (non-Colab) training script. Batches are
  daily graph snapshots (variable node count = however many tickers have a
  valid sample that day), built from Experiment 1's exact-reproduction
  dataset. Trained end-to-end (PriceEncoder + GAT + classifier head), same
  chronological split as Experiment 1.
- `results/experiment4_gat/` — metrics.json + EXPERIMENT4_GAT.md, following
  the Experiment 1-3 convention, explicitly reporting comparison against
  both Experiment 1's baseline and Experiment 3's price-only result.

Out of scope (explicitly deferred):
- Correlation edges, dynamic tweet-co-occurrence edges (next phase, see
  above).
- Text/tweet branch fusion with the GAT (this is price-only, matching the
  ablation row we're loosely targeting).
- Neo4j as a live data source for training (visualization only, per user
  decision).
- Event-conditioned attention.

## Data Flow

```
Experiment 1's exact-reproduction samples (26,619 ticker-date rows)
        |
Group by main_target_date -> one graph snapshot per trading day
        |
Snapshot = { tickers present that day, each with a 5-day price window }
        |
Per-ticker: PriceEncoder (GRU + temporal attention, from Exp. 3) -> 64-d
        |
Stack into one day's node-feature matrix
        |
Sector graph filtered to only the tickers present that day (edge_index)
        |
2-layer GATConv (4 heads) over that day's subgraph
        |
Per-ticker classifier head -> up/down logits
        |
Loss computed over all tickers present that day, summed/averaged per batch
```

Separately, once (not per-epoch): the full static 88-node sector graph is
pushed to Neo4j for browsing — this path never touches the training loop.

## Neo4j Setup (Aura, cloud free tier)

- User provisions a free instance at https://console.neo4j.io (Google SSO
  login to the console; the Bolt driver itself authenticates separately via
  a generated connection URI + password, not the Google credentials).
- `scripts/export_graph_to_neo4j.py` reads credentials from environment
  variables (`NEO4J_URI` — `neo4j+s://xxxxxxxx.databases.neo4j.io`,
  `NEO4J_USER` — default `neo4j`, `NEO4J_PASSWORD`) — never hardcoded, never
  committed. A `.env.example` (no real values) documents the three variables;
  the real `.env` stays gitignored.
- Schema: `(:Ticker {symbol, sector, name})-[:SAME_SECTOR]-(:Ticker)`. Script
  is idempotent (`MERGE`, not `CREATE`) so re-running it doesn't duplicate
  nodes/edges.
- Node count 88, sector edges within a 5-13-ticker sector are a small
  complete subgraph each — total edges are modest (well under 1,000),
  comfortably within Aura Free's node/relationship limits.
- Only ticker symbols, names, and sector labels are pushed — no price data,
  no tweet text, no fundamentals — so there's no sensitive-data concern in
  sending this to a cloud-hosted instance.

## Model Details

- `PriceEncoder`: reused unmodified from Experiment 3
  (`src/models/temporal_encoder.py` or wherever it currently lives in the
  MAN-SF notebook — will be extracted into `src/models/` as a proper module
  during implementation, since it's now used by two experiments).
- `StockGAT`: `torch_geometric.nn.GATConv(64, 64, heads=4, concat=False)` x2,
  ReLU + dropout between layers, followed by a linear classifier head
  (64 -> 32 -> 2) per node.
- No text branch, no fundamentals branch in this experiment (price-only,
  matching the ablation row this loosely targets).

## Training Loop

- One epoch = iterate over all trading days in the train split in
  chronological order (no shuffling across days, since each day's graph
  structure and node set differs — shuffling *within* a day's node order is
  fine and harmless).
- Each day's forward/backward pass uses that day's `torch_geometric.data.Data`
  object (variable node count, sector-filtered edge_index).
- Optimizer: Adam. Early stopping on validation MCC, patience matching the
  project convention (10, per `src/training/trainer.py`).
- Runs locally (CPU, no GPU) — 88 nodes/day, ~500 trading days, is a small
  enough computation that this is expected to be fast (seconds-minutes per
  epoch, not hours).

## Dependencies / Risks

- `torch_geometric` is not yet installed. Will attempt `pip install
  torch_geometric` matched to the existing `torch==2.8.0+cpu` install.
  **Fallback if the install is broken/version-mismatched**: implement the
  GAT attention mechanism manually in plain PyTorch (small graphs, fully
  feasible) rather than blocking on the dependency.
- `neo4j` Python driver: `pip install neo4j`.
- Requires the user to have Neo4j Desktop/Docker running locally before
  `export_graph_to_neo4j.py` will succeed — this is a manual one-time setup
  step outside this codebase's control.

## Evaluation / Reporting

`results/experiment4_gat/EXPERIMENT4_GAT.md` reports, following Experiments
1-3's convention: Accuracy/F1/MCC/AUC on the test split, compared explicitly
against Experiment 1 (+0.092) and Experiment 3 price-only (+0.036), plus the
"Reproduction vs. Contribution" framing above so the paper doesn't
overclaim vs. MAN-SF's actual (Wikidata-graph) 0.117 figure.
