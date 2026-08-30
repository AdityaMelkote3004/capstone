# Graph Transformer (Global Attention) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Experiment 5 — replace GAT's local, neighbor-restricted attention with global self-attention (every ticker attends to every other ticker in a day's cross-section) plus a learned additive structural bias for same-sector pairs, and compare the real result against Experiments 1, 3, and 4.

**Architecture:** The existing `PriceEncoder` (unchanged) still encodes each ticker's price window independently. Instead of `StockGAT`'s 2-layer `GATConv` over a sparse same-sector edge list, `StockGraphTransformer` runs 2 `nn.TransformerEncoderLayer`s with dense (all-pairs) attention, injecting a learned scalar bias for same-sector pairs via each layer's float `src_mask` argument — which PyTorch adds directly to raw attention scores before the softmax. This is additive to Experiment 4, not a replacement: `StockGAT` and its results are untouched.

**Tech Stack:** PyTorch's `nn.TransformerEncoderLayer` (core, no new dependency), reusing `networkx`, `pandas`, `torch_geometric` (only for `build_sector_graph`, not for the model itself), pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-graph-transformer-design.md`

## Global Constraints

- Never describe results as reproducing or beating MAN-SF's GAT+price MCC of 0.117 — same non-reproduction framing as Experiment 4 applies here (sector graph is a proxy, not MAN-SF's Wikidata-relation graph), regardless of attention mechanism.
- `results/experiment5_graph_transformer/metrics.json` must match the exact schema used by Experiments 1/3/4: `{"test": {"label", "backend", "accuracy", "f1", "mcc", "auc", "n_samples", "best_epoch", "best_val_mcc"}}`.
- `StockGAT`, `src/models/graph_model.py`, and `results/experiment4_gat/` must not be modified — this is a new, additive experiment for side-by-side comparison.
- Report the real result honestly against Experiments 1 (+0.0922), 3 price-only (+0.0108), and 4 sector-GAT (-0.0054) — no rounding a marginal or negative result up, matching this project's established convention.
- Follow existing repo style: module-level docstring, type hints on public functions (see `src/models/graph_model.py` for the reference style this mirrors).

---

## Task 1: `build_sector_bias` + `StockGraphTransformer` model

**Files:**
- Create: `src/models/graph_transformer.py`
- Test: `tests/test_graph_transformer.py`

**Interfaces:**
- Consumes: `PriceEncoder(input_dim: int = 3, hidden_dim: int = 64)` with `.forward(x: Tensor[N,T,input_dim], mask: Tensor[N,T]) -> Tensor[N,hidden_dim]` from `src.models.price_encoder`. `build_sector_graph(sector_csv_path: str) -> networkx.Graph` from `src.models.graph_model` (nodes are ticker symbols, `graph.has_edge(a, b)` is `True` iff `a`/`b` share a sector).
- Produces: `build_sector_bias(tickers: list, graph: nx.Graph) -> torch.Tensor` shape `(N, N)`, entries `1.0` where `tickers[i]`/`tickers[j]` share a sector (`i != j`), else `0.0` (including the diagonal). `StockGraphTransformer(price_input_dim: int = 3, hidden_dim: int = 64, nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.2)`, a `torch.nn.Module` with `.forward(price_seq: Tensor[N,T,price_input_dim], price_mask: Tensor[N,T], same_sector_mask: Tensor[N,N]) -> Tensor[N,2]`. Both consumed by Task 2's training script.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph_transformer.py`:

```python
from itertools import combinations

import torch

from src.models.graph_model import build_sector_graph
from src.models.graph_transformer import StockGraphTransformer, build_sector_bias


def _write_sector_csv(tmp_path):
    path = tmp_path / "sectors.csv"
    path.write_text("Ticker,Sector\nAAA,Tech\nBBB,Tech\nCCC,Energy\nDDD,Energy\n")
    return str(path)


def test_build_sector_bias_marks_same_sector_pairs_only(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    mask = build_sector_bias(["AAA", "BBB", "CCC"], graph)
    assert mask.shape == (3, 3)
    assert mask[0, 1].item() == 1.0 and mask[1, 0].item() == 1.0  # AAA, BBB: same sector
    assert mask[0, 2].item() == 0.0 and mask[1, 2].item() == 0.0  # different sector
    assert mask[0, 0].item() == 0.0  # diagonal always zero


def test_build_sector_bias_symmetric(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    mask = build_sector_bias(["AAA", "BBB", "CCC", "DDD"], graph)
    assert torch.equal(mask, mask.t())


def test_build_sector_bias_zero_when_no_shared_sector(tmp_path):
    graph = build_sector_graph(_write_sector_csv(tmp_path))
    mask = build_sector_bias(["AAA", "CCC"], graph)
    assert torch.equal(mask, torch.zeros(2, 2))


def test_stock_graph_transformer_forward_shape():
    model = StockGraphTransformer(price_input_dim=3, hidden_dim=16, nhead=2, dim_feedforward=32)
    price_seq = torch.randn(5, 4, 3)
    price_mask = torch.ones(5, 4)
    same_sector_mask = torch.zeros(5, 5)
    logits = model(price_seq, price_mask, same_sector_mask)
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_stock_graph_transformer_bias_changes_output():
    """The learned sector-bias parameter must actually influence attention:
    a non-zero same_sector_mask must produce different output than an
    all-zero one, given the same inputs and weights."""
    torch.manual_seed(0)
    model = StockGraphTransformer(price_input_dim=3, hidden_dim=16, nhead=2, dim_feedforward=32)
    model.eval()
    # Force the learned bias scalar away from its near-zero init so the
    # test isn't relying on a lucky random initialization.
    with torch.no_grad():
        model.sector_bias.fill_(5.0)
    price_seq = torch.randn(5, 4, 3)
    price_mask = torch.ones(5, 4)
    zero_mask = torch.zeros(5, 5)
    biased_mask = torch.zeros(5, 5)
    biased_mask[0, 1] = biased_mask[1, 0] = 1.0
    with torch.no_grad():
        out_zero = model(price_seq, price_mask, zero_mask)
        out_biased = model(price_seq, price_mask, biased_mask)
    assert not torch.allclose(out_zero, out_biased)


def test_stock_graph_transformer_does_not_collapse_after_training():
    """Global attention can still over-smooth with enough layers/training,
    just less severely than GAT's local message passing over a complete
    graph (see Experiment 4's diagnosed bug). This regression test trains
    briefly on a synthetic all-same-sector clique and asserts node outputs
    don't collapse to a shared constant."""
    torch.manual_seed(0)
    n = 8
    model = StockGraphTransformer(price_input_dim=3, hidden_dim=16, nhead=2, dim_feedforward=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    price_seq = torch.randn(n, 4, 3)
    price_mask = torch.ones(n, 4)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    same_sector_mask = torch.ones(n, n) - torch.eye(n)

    for _ in range(30):
        logits = model(price_seq, price_mask, same_sector_mask)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(price_seq, price_mask, same_sector_mask)
    assert logits.std(dim=0).mean().item() > 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_transformer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.graph_transformer'`

- [ ] **Step 3: Write the implementation**

Create `src/models/graph_transformer.py`:

```python
"""Graph transformer (global attention) model for Experiment 5.

Replaces GAT's local, neighbor-restricted attention (src/models/graph_model.py,
Experiment 4) with global self-attention: every ticker attends to every
other ticker in a day's cross-section, not just same-sector ones. Graph
structure (currently: same-sector membership) is injected as a learned
additive bias to raw attention scores, matching the mechanism used by
Graphormer-style graph transformers -- see
docs/superpowers/specs/2026-08-30-graph-transformer-design.md for why this
sidesteps the over-smoothing bug diagnosed in Experiment 4 (that bug came
from stacking *local* neighbor-averaging steps over a complete graph;
global attention has no such restriction to compound).

This is additive to Experiment 4, not a replacement -- StockGAT and its
results are untouched.
"""

import networkx as nx
import torch
import torch.nn as nn

from src.models.price_encoder import PriceEncoder


def build_sector_bias(tickers: list, graph: nx.Graph) -> torch.Tensor:
    """Builds a dense (N, N) structural mask for `tickers`: entry [i, j]
    is 1.0 if tickers[i] and tickers[j] share a sector (per `graph`,
    built by src.models.graph_model.build_sector_graph) and i != j, else
    0.0. The diagonal is always 0.0 -- self-attention is already handled
    natively by nn.TransformerEncoderLayer, no separate self bias needed.
    This mask is multiplied by a single learned scalar inside
    StockGraphTransformer to produce the actual additive attention bias."""
    n = len(tickers)
    mask = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j and graph.has_edge(tickers[i], tickers[j]):
                mask[i, j] = 1.0
    return mask


class StockGraphTransformer(nn.Module):
    """Price-only graph transformer: PriceEncoder per ticker -> 2-layer
    global self-attention (nn.TransformerEncoderLayer) with a learned
    same-sector attention bias -> per-ticker binary classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.sector_bias = nn.Parameter(torch.zeros(1))
        layer_kwargs = dict(d_model=hidden_dim, nhead=nhead,
                             dim_feedforward=dim_feedforward, dropout=dropout,
                             batch_first=True)
        self.layer1 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.layer2 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                same_sector_mask: torch.Tensor) -> torch.Tensor:
        node_features = self.price_encoder(price_seq, price_mask)  # (N, hidden_dim)
        bias = same_sector_mask * self.sector_bias  # (N, N), learned scalar at same-sector pairs
        x = node_features.unsqueeze(0)  # (1, N, hidden_dim) for batch_first Transformer layers
        x = self.layer1(x, src_mask=bias)
        x = self.layer2(x, src_mask=bias)
        h = x.squeeze(0)  # (N, hidden_dim)
        return self.classifier(h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_transformer.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/graph_transformer.py tests/test_graph_transformer.py
git commit -m "Add StockGraphTransformer: global attention + sector bias (Experiment 5)"
```

---

## Task 2: Training script and Experiment 5 results

**Files:**
- Create: `scripts/train_graph_transformer.py`
- Create (after running the script): `results/experiment5_graph_transformer/metrics.json`, `results/experiment5_graph_transformer/EXPERIMENT5_GRAPH_TRANSFORMER.md`

**Interfaces:**
- Consumes: `build_daily_snapshots`, `normalize_price_snapshots` from `src.data.graph_dataset` (unchanged, reused from Experiment 4). `build_sector_graph` from `src.models.graph_model` (unchanged, reused). `build_sector_bias`, `StockGraphTransformer` from Task 1. `set_seed` from `src.utils.seed`.
- Produces: `results/experiment5_graph_transformer/metrics.json` matching the schema `{"test": {"label", "backend", "accuracy", "f1", "mcc", "auc", "n_samples", "best_epoch", "best_val_mcc"}}`.

- [ ] **Step 1: Write the training script**

Create `scripts/train_graph_transformer.py`:

```python
"""Trains the graph transformer (Experiment 5): PriceEncoder + 2-layer
global self-attention with a learned same-sector bias, one gradient step
per trading-day snapshot. Reports Accuracy/F1/MCC/AUC on the exact-
reproduction test split, in the same schema as Experiments 1, 3, and 4.

Reminder before citing results: this remains a sector-based proxy graph,
not MAN-SF's Wikidata-relation graph, regardless of attention mechanism --
see docs/superpowers/specs/2026-08-30-graph-transformer-design.md.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots
from src.models.graph_model import build_sector_graph
from src.models.graph_transformer import StockGraphTransformer, build_sector_bias
from src.utils.seed import set_seed

PARQUET = "dataset/stocknet_protocol_a_exact.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
OUT_DIR = "results/experiment5_graph_transformer"
PATIENCE = 10
MAX_EPOCHS = 100
LR = 1e-3


def run_epoch(model, snapshots, graph, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        snapshots = list(snapshots)
        random.shuffle(snapshots)
    all_probs, all_targets = [], []
    total_loss = 0.0
    for snap in snapshots:
        same_sector_mask = build_sector_bias(snap.tickers, graph)
        logits = model(snap.price_seq, snap.price_mask, same_sector_mask)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    normalize_price_snapshots(train_snaps, dev_snaps, test_snaps)
    n_samples = sum(len(s.tickers) for s in test_snaps)

    model = StockGraphTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_mcc = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        train_loss, train_metrics = run_epoch(model, train_snaps, graph, optimizer)
        _, val_metrics = run_epoch(model, dev_snaps, graph)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"train_mcc={train_metrics['mcc']:.4f} val_mcc={val_metrics['mcc']:.4f}")
        if val_metrics["mcc"] > best_val_mcc:
            best_val_mcc = val_metrics["mcc"]
            best_epoch = epoch
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
            {"test": {
                "label": "Graph transformer, global attention (price only)",
                "backend": None,
                **test_metrics,
                "n_samples": n_samples,
                "best_epoch": best_epoch,
                "best_val_mcc": best_val_mcc,
            }},
            f, indent=2,
        )
    print(f"Saved {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/train_graph_transformer.py`

Expected: prints per-epoch train/val MCC, stops early or at 100 epochs, prints final test metrics, writes `results/experiment5_graph_transformer/metrics.json`. Read the actual printed test MCC/F1/Accuracy/AUC values — they are needed verbatim for Step 3. Runs locally on CPU in minutes (dense attention over ≤90 nodes/day is trivially cheap).

- [ ] **Step 3: Write the results doc**

Create `results/experiment5_graph_transformer/EXPERIMENT5_GRAPH_TRANSFORMER.md`, following the structure of `results/experiment4_gat/EXPERIMENT4_GAT.md`. Use this skeleton, filling the bracketed values with the actual numbers from Step 2's `metrics.json` (do not invent numbers — copy them from the real run):

```markdown
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
| **Experiment 5 graph transformer (price only)** | [accuracy] | [f1] | [mcc] | [auc] |

[One paragraph: state plainly whether this beats, matches, or still
underperforms Experiments 3/4/1. Do not round a marginal or negative
result up. If it improves on Experiment 4, say by how much and whether it
closes the gap to Experiment 3 or Experiment 1. If it doesn't improve,
say so and note that Experiment 4's over-smoothing bug being fixed
doesn't guarantee global attention finds useful signal in a sector-only
graph -- these are separate questions.]

## What this experiment does and doesn't establish

Establishes: whether global attention (with the over-smoothing mechanism
architecturally avoided) can find useful signal in same-sector structure
that local GAT message-passing could not, once GAT's own bug is fixed.

Does not establish: whether a sparser or different graph (correlation
edges, dynamic tweet-co-occurrence edges, or MAN-SF's own Wikidata
relations) would perform differently under either attention mechanism --
the edge definition (same-sector) is held constant across Experiments 4
and 5 specifically so the comparison isolates the attention mechanism,
not the graph.

## How we should move forward

[Fill in based on the actual result: if Experiment 5 beats Experiment 4
but still trails Experiments 1/3, the natural next step is combining
global attention with a richer edge type (correlation/dynamic) rather
than more attention-mechanism tuning. If Experiment 5 doesn't improve on
Experiment 4 either, that's evidence the sector-only graph itself (not
the attention mechanism) is the limiting factor, strengthening the case
for building correlation/dynamic edges next per
`results/experiment4_gat/RELATED_WORK.md`.]

## Reproducing this experiment

```bash
python scripts/train_graph_transformer.py
```

Runs locally on CPU (no GPU required) -- dense attention over at most ~90
tickers/day is inexpensive, completes in minutes.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_graph_transformer.py results/experiment5_graph_transformer/
git commit -m "Add Experiment 5: graph transformer training script and results"
```
