# Sector-Graph GAT + Neo4j Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Experiment 4 — a sector-graph GAT that lets each ticker's price representation attend to same-sector neighbors — plus a one-time Neo4j visualization of that graph, and report results against Experiments 1 and 3.

**Architecture:** A shared `PriceEncoder` (GRU + temporal attention, extracted from Experiment 3) encodes each ticker's 5-day price window independently; a 2-layer `GATConv` then updates each ticker's embedding by attending to same-sector neighbors within that trading day's cross-section; a linear head classifies up/down per ticker. Training iterates one trading day at a time (one graph snapshot = one gradient step), using the same chronological train/dev/test split already baked into `dataset/stocknet_protocol_a_exact.parquet`. Neo4j Aura receives the static 87-node sector graph once, for browsing only — it is never read during training.

**Tech Stack:** PyTorch 2.11 (CPU), torch_geometric 2.8 (`GATConv`), networkx, neo4j Python driver 6.3, python-dotenv, pandas, scikit-learn (metrics), pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md`

## Global Constraints

- Never describe this experiment's results as reproducing or beating MAN-SF's GAT+price MCC of 0.117 — that figure was measured on a Wikidata company-relation graph, not a sector graph (verified against `aclanthology.org/2020.emnlp-main.676`). Report this experiment's own numbers on their own terms.
- `PriceEncoder` is a reimplementation of a component from Sawhney et al. (EMNLP 2020, MAN-SF), built from the paper's architecture description — any docstring/comment referencing it must say "reused from Experiment 3's MAN-SF reimplementation," never imply original invention.
- Neo4j credentials (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) come only from environment variables loaded via `.env` (already created locally, already git-ignored). Never hardcode, print, or commit them.
- Results go in `results/experiment4_gat/` with a `metrics.json` (same schema as Experiments 1/3: `{"test": {"label", "backend", "accuracy", "f1", "mcc", "auc"}}`) and an `EXPERIMENT4_GAT.md` write-up, matching the existing convention.
- Existing dataset files are the source of truth and must not be modified: `dataset/stocknet_protocol_a_exact.parquet` (26,619 rows, columns `Ticker`, `Target_Date`, `Split` ∈ {train, dev, test}, `Target`, `Window_Length`, `Day{1-4}_Price{1-3}`) and `dataset/final/sector_mapping.csv` (87 tickers, `Ticker`, `Sector` columns, 9 sectors).
- Follow existing repo style: module-level docstring, type hints on public functions, no framework-mandated boilerplate beyond that (see `src/models/baselines.py` for the reference style).

---

## Task 1: Add and verify new dependencies

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `torch_geometric`, `neo4j`, `python-dotenv` importable in the environment for all later tasks.

- [ ] **Step 1: Add the three new dependencies**

Add to `requirements.txt` (after the existing `pyyaml>=6.0` line, before the "Interactive dashboard" section):

```
# Graph neural network + Neo4j visualization (Experiment 4)
torch_geometric>=2.8.0
neo4j>=6.3.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: Verify installation**

Run: `pip install -r requirements.txt`

Then run this smoke check:

```bash
python -c "
import torch
from torch_geometric.nn import GATConv
from neo4j import GraphDatabase
import dotenv
x = torch.randn(5, 8)
edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]])
out = GATConv(8, 4, heads=2, concat=False)(x, edge_index)
assert out.shape == (5, 4)
print('OK')
"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "Add torch_geometric, neo4j, python-dotenv for Experiment 4"
```

---

## Task 2: Extract `PriceEncoder` into `src/models/price_encoder.py`

**Files:**
- Create: `src/models/price_encoder.py`
- Test: `tests/test_price_encoder.py`

**Interfaces:**
- Produces: `PriceEncoder(input_dim: int = 3, hidden_dim: int = 64)`, a `torch.nn.Module` with `.forward(x: Tensor[batch, T, input_dim], mask: Tensor[batch, T]) -> Tensor[batch, hidden_dim]` and attribute `.output_dim`. Consumed by `StockGAT` in Task 3.

This is a verbatim extraction of the `PriceEncoder` class already defined and validated in `preprocessing_notebooks/MMGTFFF_mansf_joint_bilinear.ipynb` (cell defining `class PriceEncoder`), given a proper module home since Experiments 3 and 4 now both depend on it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_encoder.py`:

```python
import torch

from src.models.price_encoder import PriceEncoder


def test_output_shape():
    encoder = PriceEncoder(input_dim=3, hidden_dim=64)
    x = torch.randn(5, 4, 3)
    mask = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.float32,
    )
    out = encoder(x, mask)
    assert out.shape == (5, 64)


def test_handles_single_valid_day_without_nan():
    encoder = PriceEncoder(input_dim=3, hidden_dim=64)
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    out = encoder(x, mask)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_price_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.price_encoder'`

- [ ] **Step 3: Write the implementation**

Create `src/models/price_encoder.py`:

```python
"""GRU + temporal-attention price encoder.

Reused unmodified from Experiment 3's MAN-SF reimplementation
(preprocessing_notebooks/MMGTFFF_mansf_joint_bilinear.ipynb), which itself
is built from the architecture description in Sawhney et al. (EMNLP 2020,
"Deep Attentive Learning for Stock Movement Prediction From Social Media
Text and Company Correlations") -- given its own module here since
Experiments 3 and 4 both depend on it.
"""

import torch
import torch.nn as nn


class PriceEncoder(nn.Module):
    """Encodes a masked, variable-length price window into a fixed vector."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.temporal_score = nn.Linear(hidden_dim, 1)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, input_dim); mask: (batch, T), 1 = real day, 0 = padding
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, total_length=x.shape[1]
        )
        scores = self.temporal_score(gru_out).squeeze(-1).masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * gru_out).sum(dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_price_encoder.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/price_encoder.py tests/test_price_encoder.py
git commit -m "Extract PriceEncoder into its own module for reuse in Experiment 4"
```

---

## Task 3: Sector graph construction + `StockGAT` model

**Files:**
- Create: `src/models/graph_model.py`
- Test: `tests/test_graph_model.py`

**Interfaces:**
- Consumes: `PriceEncoder` from Task 2 (`src.models.price_encoder.PriceEncoder`).
- Produces:
  - `build_sector_graph(sector_csv_path: str) -> networkx.Graph` — nodes are ticker symbols with `sector`/`name` attributes, edges connect same-sector tickers.
  - `edge_index_for_tickers(graph: networkx.Graph, tickers: list[str]) -> torch.Tensor` shape `(2, E)`, local indices `0..len(tickers)-1` matching the order of `tickers`. Consumed by Task 4's dataset collation and Task 6's training script.
  - `StockGAT(price_input_dim: int = 3, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.2)`, a `torch.nn.Module` with `.forward(price_seq: Tensor[N, T, price_input_dim], price_mask: Tensor[N, T], edge_index: Tensor[2, E]) -> Tensor[N, 2]` (binary logits per node).

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph_model.py`:

```python
import torch

from src.models.graph_model import StockGAT, build_sector_graph, edge_index_for_tickers


def _write_sector_csv(tmp_path):
    path = tmp_path / "sectors.csv"
    path.write_text("Ticker,Sector\nAAA,Tech\nBBB,Tech\nCCC,Energy\nDDD,Energy\n")
    return str(path)


def test_build_sector_graph_connects_same_sector_only(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    assert graph.number_of_nodes() == 4
    assert graph.has_edge("AAA", "BBB")
    assert graph.has_edge("CCC", "DDD")
    assert not graph.has_edge("AAA", "CCC")
    assert graph.nodes["AAA"]["sector"] == "Tech"


def test_edge_index_for_tickers_uses_local_indices(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    edge_index = edge_index_for_tickers(graph, ["AAA", "BBB", "CCC"])
    edge_set = set(map(tuple, edge_index.t().tolist()))
    assert (0, 1) in edge_set and (1, 0) in edge_set
    assert (0, 2) not in edge_set and (1, 2) not in edge_set


def test_edge_index_for_tickers_empty_when_no_shared_sector(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    edge_index = edge_index_for_tickers(graph, ["AAA", "CCC"])
    assert edge_index.shape == (2, 0)


def test_stock_gat_forward_shape():
    model = StockGAT(price_input_dim=3, hidden_dim=16, heads=2)
    price_seq = torch.randn(4, 4, 3)
    price_mask = torch.ones(4, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    logits = model(price_seq, price_mask, edge_index)
    assert logits.shape == (4, 2)


def test_stock_gat_forward_with_no_edges():
    model = StockGAT(price_input_dim=3, hidden_dim=16, heads=2)
    price_seq = torch.randn(3, 4, 3)
    price_mask = torch.ones(3, 4)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    logits = model(price_seq, price_mask, edge_index)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.graph_model'`

- [ ] **Step 3: Write the implementation**

Create `src/models/graph_model.py`:

```python
"""Sector graph construction and the sector-graph GAT model (Experiment 4).

The sector graph connects tickers that share a `Sector` label -- a cheaper,
materially different proxy for MAN-SF's Wikidata company-relation graph,
NOT a reproduction of it. See
docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md for why
this experiment's results must not be compared directly to MAN-SF's
GAT+price ablation figure (0.117 MCC, measured on their Wikidata graph).
"""

from itertools import combinations

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from src.models.price_encoder import PriceEncoder


def build_sector_graph(sector_csv_path: str) -> nx.Graph:
    """Builds an undirected graph: one node per ticker, edges between
    tickers in the same sector. Node attributes: sector, name."""
    df = pd.read_csv(sector_csv_path)
    graph = nx.Graph()
    for _, row in df.iterrows():
        graph.add_node(row["Ticker"], sector=row["Sector"], name=row["Ticker"])
    for _, group in df.groupby("Sector"):
        for ticker_a, ticker_b in combinations(sorted(group["Ticker"]), 2):
            graph.add_edge(ticker_a, ticker_b)
    return graph


def edge_index_for_tickers(graph: nx.Graph, tickers: list) -> torch.Tensor:
    """Builds a (2, E) edge_index over `tickers`, using each ticker's
    position in the list as its node index, restricted to `graph` edges
    where both endpoints are in `tickers`. Returns both directions per
    undirected edge (required by GATConv's message passing)."""
    index_of = {ticker: i for i, ticker in enumerate(tickers)}
    ticker_set = set(tickers)
    src, dst = [], []
    for a, b in graph.edges():
        if a in ticker_set and b in ticker_set:
            src.extend([index_of[a], index_of[b]])
            dst.extend([index_of[b], index_of[a]])
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


class StockGAT(nn.Module):
    """Price-only sector-graph GAT: PriceEncoder per ticker -> 2-layer
    GATConv over the sector graph -> per-ticker binary classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=False, dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads,
                             concat=False, dropout=dropout, add_self_loops=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        node_features = self.price_encoder(price_seq, price_mask)
        h = torch.relu(self.gat1(node_features, edge_index))
        h = self.dropout(h)
        h = torch.relu(self.gat2(h, edge_index))
        return self.classifier(h)
```

Note: `add_self_loops=True` (GATConv's default) is what keeps `test_stock_gat_forward_with_no_edges` finite — an isolated node still attends to itself.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_model.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/graph_model.py tests/test_graph_model.py
git commit -m "Add sector graph construction and StockGAT model"
```

---

## Task 4: Daily graph-snapshot dataset builder

**Files:**
- Create: `src/data/graph_dataset.py`
- Test: `tests/test_graph_dataset.py`

**Interfaces:**
- Consumes: `edge_index_for_tickers` from Task 3 (`src.models.graph_model.edge_index_for_tickers`), a `networkx.Graph` from `build_sector_graph`.
- Produces:
  - `DaySnapshot` dataclass with fields `tickers: list[str]`, `price_seq: Tensor[N, 4, 3]`, `price_mask: Tensor[N, 4]`, `target: Tensor[N]`.
  - `build_daily_snapshots(df: pandas.DataFrame, split: str) -> list[DaySnapshot]`, sorted chronologically by `Target_Date`. Consumed by Task 6's training script.

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph_dataset.py`:

```python
import pandas as pd
import torch

from src.data.graph_dataset import build_daily_snapshots
from src.models.graph_model import build_sector_graph, edge_index_for_tickers


def _make_df():
    rows = []
    for ticker, date, split, window_length, target in [
        ("AAA", "2020-01-01", "train", 2, 1),
        ("BBB", "2020-01-01", "train", 2, 0),
        ("AAA", "2020-01-02", "train", 3, 0),
        ("CCC", "2020-01-02", "train", 1, 1),
        ("BBB", "2020-01-03", "dev", 4, 1),
    ]:
        row = {
            "Ticker": ticker, "Target_Date": date, "Split": split,
            "Window_Length": window_length, "Target": target,
        }
        for d in range(1, 5):
            for p in range(1, 4):
                row[f"Day{d}_Price{p}"] = 0.1 * d + 0.01 * p
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_daily_snapshots_groups_by_date_within_split():
    snapshots = build_daily_snapshots(_make_df(), "train")
    assert len(snapshots) == 2
    assert snapshots[0].tickers == ["AAA", "BBB"]
    assert snapshots[1].tickers == ["AAA", "CCC"]
    assert snapshots[0].price_seq.shape == (2, 4, 3)
    assert snapshots[0].target.tolist() == [1, 0]


def test_build_daily_snapshots_respects_window_length_mask():
    snapshots = build_daily_snapshots(_make_df(), "train")
    # AAA on 2020-01-01 has Window_Length=2 -> first two days real, rest padding.
    assert snapshots[0].price_mask[0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_build_daily_snapshots_filters_to_requested_split():
    snapshots = build_daily_snapshots(_make_df(), "dev")
    assert len(snapshots) == 1
    assert snapshots[0].tickers == ["BBB"]


def test_snapshot_edge_index_matches_sector_graph(tmp_path):
    sector_csv = tmp_path / "sectors.csv"
    sector_csv.write_text("Ticker,Sector\nAAA,Tech\nBBB,Tech\nCCC,Energy\n")
    graph = build_sector_graph(str(sector_csv))
    snapshots = build_daily_snapshots(_make_df(), "train")
    day1_edges = edge_index_for_tickers(graph, snapshots[0].tickers)
    day2_edges = edge_index_for_tickers(graph, snapshots[1].tickers)
    assert day1_edges.shape[1] == 2  # AAA<->BBB, both directions
    assert day2_edges.shape[1] == 0  # AAA (Tech) and CCC (Energy) don't connect
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.graph_dataset'`

- [ ] **Step 3: Write the implementation**

Create `src/data/graph_dataset.py`:

```python
"""Builds per-trading-day graph snapshots for Experiment 4's sector-graph
GAT: each snapshot is the cross-section of tickers with a valid sample on
one Target_Date, the unit the GAT attends over. Reuses
dataset/stocknet_protocol_a_exact.parquet's existing chronological
train/dev/test split -- no new splitting logic here.
"""

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass
class DaySnapshot:
    tickers: list
    price_seq: torch.Tensor   # (N, 4, 3)
    price_mask: torch.Tensor  # (N, 4)
    target: torch.Tensor      # (N,)


def _row_to_price_seq(row: pd.Series) -> torch.Tensor:
    seq = torch.zeros(4, 3)
    for d in range(4):
        for p in range(3):
            seq[d, p] = row[f"Day{d + 1}_Price{p + 1}"]
    return seq


def _window_mask(window_length: int) -> torch.Tensor:
    mask = torch.zeros(4)
    mask[: min(int(window_length), 4)] = 1.0
    return mask


def build_daily_snapshots(df: pd.DataFrame, split: str) -> list:
    """Groups rows of `df` where Split == split by Target_Date, sorted
    chronologically, returning one DaySnapshot per date."""
    subset = df[df["Split"] == split].sort_values("Target_Date")
    snapshots = []
    for _, day_df in subset.groupby("Target_Date", sort=True):
        tickers = day_df["Ticker"].tolist()
        price_seq = torch.stack([_row_to_price_seq(row) for _, row in day_df.iterrows()])
        price_mask = torch.stack([_window_mask(wl) for wl in day_df["Window_Length"]])
        target = torch.tensor(day_df["Target"].values, dtype=torch.long)
        snapshots.append(DaySnapshot(tickers=tickers, price_seq=price_seq,
                                      price_mask=price_mask, target=target))
    return snapshots
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_dataset.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/graph_dataset.py tests/test_graph_dataset.py
git commit -m "Add daily graph-snapshot dataset builder for Experiment 4"
```

---

## Task 5: Verify the full pipeline against the real dataset (no unit test — integration check)

**Files:**
- None created; this is a verification step before writing the training script, following this project's established "validate against real data before shipping" pattern.

**Interfaces:**
- None new — exercises Tasks 2-4 together against real data.

- [ ] **Step 1: Run an end-to-end shape/sanity check against the real files**

```bash
python -c "
import pandas as pd
import torch

from src.data.graph_dataset import build_daily_snapshots
from src.models.graph_model import build_sector_graph, edge_index_for_tickers, StockGAT

df = pd.read_parquet('dataset/stocknet_protocol_a_exact.parquet')
graph = build_sector_graph('dataset/final/sector_mapping.csv')
assert graph.number_of_nodes() == 87

train_snaps = build_daily_snapshots(df, 'train')
dev_snaps = build_daily_snapshots(df, 'dev')
test_snaps = build_daily_snapshots(df, 'test')
print('snapshots:', len(train_snaps), len(dev_snaps), len(test_snaps))
assert len(train_snaps) + len(dev_snaps) + len(test_snaps) == df['Target_Date'].nunique()

model = StockGAT()
snap = train_snaps[0]
edge_index = edge_index_for_tickers(graph, snap.tickers)
logits = model(snap.price_seq, snap.price_mask, edge_index)
assert logits.shape == (len(snap.tickers), 2)
assert torch.isfinite(logits).all()
print('OK -- pipeline runs end to end on real data')
"
```

Expected: prints snapshot counts (should sum to 504, the number of unique `Target_Date` values) and `OK -- pipeline runs end to end on real data`.

- [ ] **Step 2: If this fails, stop and fix the root cause before proceeding**

Do not proceed to Task 6 until this passes — it's the cheapest point to catch a data-shape mismatch, before wiring up the full training loop.

No commit for this task (no files changed).

---

## Task 6: Neo4j export script

**Files:**
- Create: `scripts/export_graph_to_neo4j.py`

**Interfaces:**
- Consumes: `build_sector_graph` from Task 3.
- Produces: pushes the static sector graph to the user's live Neo4j Aura instance. Not consumed by any other code — visualization only, per the spec.

- [ ] **Step 1: Write the script**

Create `scripts/export_graph_to_neo4j.py`:

```python
"""One-time export of the static sector graph to Neo4j Aura, for
visualization only -- never read at training time. See
docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md.

Requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in a local .env (see
.env.example for the expected keys). Credentials are read from the
environment only -- never hardcoded, never logged, never committed.
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.models.graph_model import build_sector_graph

SECTOR_CSV = "dataset/final/sector_mapping.csv"


def push_graph(driver, graph) -> None:
    with driver.session() as session:
        for ticker, attrs in graph.nodes(data=True):
            session.run(
                "MERGE (t:Ticker {symbol: $symbol}) "
                "SET t.sector = $sector, t.name = $name",
                symbol=ticker, sector=attrs["sector"], name=attrs["name"],
            )
        for a, b in graph.edges():
            session.run(
                "MATCH (x:Ticker {symbol: $a}), (y:Ticker {symbol: $b}) "
                "MERGE (x)-[:SAME_SECTOR]-(y)",
                a=a, b=b,
            )


def main() -> None:
    load_dotenv()
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]

    graph = build_sector_graph(SECTOR_CSV)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        push_graph(driver, graph)
        print(f"Pushed {graph.number_of_nodes()} tickers and "
              f"{graph.number_of_edges()} SAME_SECTOR edges to Neo4j.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the live Aura instance**

Run: `python scripts/export_graph_to_neo4j.py`

Expected output: `Pushed 87 tickers and 381 SAME_SECTOR edges to Neo4j.` (381 = sum over 9 sectors of C(sector_size, 2): eight sectors of 10 tickers contribute C(10,2)=45 each = 360, plus one sector of 7 (Conglomerates) contributing C(7,2)=21 = 381 — if the printed count differs, recompute from `sector_mapping.csv`'s actual sector sizes rather than assuming a mismatch is a bug.)

- [ ] **Step 3: Verify in Neo4j Aura console**

Go to https://console.neo4j.io, open the instance, use the Query tab, and run:

```cypher
MATCH (t:Ticker) RETURN t.sector AS sector, count(*) AS n ORDER BY n DESC
```

Expected: 9 rows (one per sector), counts matching `sector_mapping.csv`'s `Sector.value_counts()` (Consumer_Goods 10, Industrial_Goods 10, Healthcare 10, Utilities 10, Services 10, Basic_Materials 10, Financial 10, Technology 10, Conglomerates 7).

Then run:

```cypher
MATCH (a:Ticker)-[:SAME_SECTOR]-(b:Ticker) RETURN a.sector AS sector, count(*) / 2 AS edges ORDER BY edges DESC
```

to visually confirm no edges cross sector boundaries, and use Aura's built-in graph view to look at the resulting clusters.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_graph_to_neo4j.py
git commit -m "Add Neo4j export script for sector graph visualization"
```

---

## Task 7: Training script and Experiment 4 results

**Files:**
- Create: `scripts/train_gat.py`
- Create (after running the script): `results/experiment4_gat/metrics.json`, `results/experiment4_gat/EXPERIMENT4_GAT.md`

**Interfaces:**
- Consumes: `build_daily_snapshots` (Task 4), `build_sector_graph`, `edge_index_for_tickers`, `StockGAT` (Task 3), `set_seed` from `src.utils.seed`.
- Produces: `results/experiment4_gat/metrics.json` matching the `{"test": {"label", "backend", "accuracy", "f1", "mcc", "auc"}}` schema used by Experiments 1 and 3.

- [ ] **Step 1: Write the training script**

Create `scripts/train_gat.py`:

```python
"""Trains the sector-graph GAT (Experiment 4): PriceEncoder + 2-layer GAT
over same-sector edges, one gradient step per trading-day snapshot.
Reports Accuracy/F1/MCC/AUC on the exact-reproduction test split, in the
same schema as Experiments 1 and 3.

Reminder before citing results: this is a labeled proxy for MAN-SF's
GAT+price component (sector edges, not their Wikidata company-relation
graph) -- see
docs/superpowers/specs/2026-08-30-sector-graph-gat-neo4j-design.md.
"""

import json
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.data.graph_dataset import build_daily_snapshots
from src.models.graph_model import StockGAT, build_sector_graph, edge_index_for_tickers
from src.utils.seed import set_seed

PARQUET = "dataset/stocknet_protocol_a_exact.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
OUT_DIR = "results/experiment4_gat"
PATIENCE = 10
MAX_EPOCHS = 100
LR = 1e-3


def run_epoch(model, snapshots, graph, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    all_probs, all_targets = [], []
    total_loss = 0.0
    for snap in snapshots:
        edge_index = edge_index_for_tickers(graph, snap.tickers)
        logits = model(snap.price_seq, snap.price_mask, edge_index)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(snap.tickers)
        all_probs.append(torch.softmax(logits, dim=-1)[:, 1].detach())
        all_targets.append(snap.target)
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "auc": roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.5,
    }
    return total_loss / len(targets), metrics


def main():
    set_seed(42)
    df = pd.read_parquet(PARQUET)
    graph = build_sector_graph(SECTOR_CSV)
    train_snaps = build_daily_snapshots(df, "train")
    dev_snaps = build_daily_snapshots(df, "dev")
    test_snaps = build_daily_snapshots(df, "test")

    model = StockGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mcc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        train_loss, train_metrics = run_epoch(model, train_snaps, graph, optimizer)
        _, val_metrics = run_epoch(model, dev_snaps, graph)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"train_mcc={train_metrics['mcc']:.4f} val_mcc={val_metrics['mcc']:.4f}")
        if val_metrics["mcc"] > best_val_mcc:
            best_val_mcc = val_metrics["mcc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    _, test_metrics = run_epoch(model, test_snaps, graph)
    print("Test metrics:", test_metrics)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(
            {"test": {"label": "Sector-graph GAT (price only)", "backend": None, **test_metrics}},
            f, indent=2,
        )
    print(f"Saved {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/train_gat.py`

Expected: prints per-epoch train/val MCC, stops early or at 100 epochs, prints final test metrics, and writes `results/experiment4_gat/metrics.json`. Read the actual printed test MCC/F1/Accuracy/AUC values — they are needed verbatim for Step 3.

- [ ] **Step 3: Write the results doc**

Create `results/experiment4_gat/EXPERIMENT4_GAT.md`, following the structure of `results/experiment3_mansf_bilinear/EXPERIMENT3_MANSF_BILINEAR.md`. Use this skeleton, filling the bracketed values with the actual numbers from Step 2's `metrics.json` (do not invent numbers — copy them from the real run):

```markdown
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
| Sector-graph GAT (price only) | [accuracy] | [f1] | [mcc] | [auc] |

Compared against:
- Experiment 1 baseline (LSTM, raw tweet count): MCC +0.092
- Experiment 3 price-only (GRU + temporal attention, no graph): MCC +0.036

[One paragraph: does the sector graph help relative to Experiment 3's
price-only result? State the real comparison plainly -- if it's better,
say by how much; if it's not, say so and don't round up.]

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
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_gat.py results/experiment4_gat/
git commit -m "Add Experiment 4: sector-graph GAT training script and results"
```

---

## Task 8: Update project-level docs to reference Experiment 4

**Files:**
- Modify: `CLAUDE.md` (only if the "Current State" section is meant to stay current — check whether this project maintains that section actively before editing)

**Interfaces:**
- None — documentation only.

- [ ] **Step 1: Check whether `CLAUDE.md`'s "Current State" section is actively maintained**

Run: `git log --oneline -5 -- CLAUDE.md`

If recent commits touch this file to reflect completed experiments, add a short note under a new "Experiment 4" heading (following the existing style of the Phase 0/1 summary) pointing to `results/experiment4_gat/EXPERIMENT4_GAT.md`. If the file hasn't been updated to track Experiments 1-3 already, skip this step — don't introduce a maintenance pattern the project hasn't been following.

- [ ] **Step 2: Commit if changed**

```bash
git add CLAUDE.md
git commit -m "Note Experiment 4 (sector-graph GAT) in project state summary"
```

(Skip this commit entirely if Step 1 concluded no edit was needed.)
