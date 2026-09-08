# Experiment 6 — Dense Graph + Model Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three leakage-safe edge types (sector, rolling-correlation, tweet co-mention), two new graph models (GraphSAGE, R-GCN), and a crash-resilient grid runner that answers whether a richer graph beats Experiment 4's sector-only graph.

**Architecture:** Edge builders live in `src/data/graph_edges.py` and return per-target-date ticker pairs, each taking an explicit time cutoff so leakage is unit-tested rather than assumed. New models join the existing `StockGAT`/`StockGraphTransformer` in `src/models/`. A grid runner script iterates `(edge_combo, model, seed, text_backend)`, writing results after every single run so a crashed Colab session resumes instead of restarting. A thin notebook clones the repo and calls the runner.

**Tech Stack:** PyTorch 2.11, torch_geometric 2.8 (`GATConv`, `TransformerConv`, `SAGEConv`, `RGCNConv`), pandas, networkx, matplotlib, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-dense-graph-experiment-design.md`

## Global Constraints

- Never describe results as reproducing or beating MAN-SF's GAT+price MCC of 0.117 — that figure was measured on a Wikidata company-relation graph, not any graph built here.
- All runs use `dataset/stocknet_protocol_a_exact.parquet` for samples/labels/splits. Test `n_samples` must remain **3720**, matching Experiments 1/3/4/5.
- `Return` and `Event_Texts` come from `dataset/final/stocknet_final_modeling_set.parquet`, joined on `(Ticker, Date)`. **17 of 26,619 exact-protocol samples are uncovered** by this join — they must degrade to empty neighbourhoods, never crash and never be dropped.
- **Leakage is a tested property.** Every edge builder takes an explicit cutoff; no edge for target date `t` may use data from `t` or later. Every builder has a test asserting this.
- Correlation defaults: `window=60`, `k=5`, `min_periods=10`. The 60-day window is measured, not arbitrary — at 20 days only 28.7% of ticker pairs have >=10 overlapping observations vs 97.7% at 60.
- Report **mean ± std across seeds**, never a single seed. Differences smaller than seed spread are "indistinguishable", not "improvements".
- Results go in `results/experiment6_dense_graph/`, `metrics.json` schema matching Experiments 1/3/4/5: `{"test": {"label", "backend", "accuracy", "f1", "mcc", "auc", "n_samples", "best_epoch", "best_val_mcc"}}`.
- Existing files must NOT be modified except where a task explicitly says so: `src/models/graph_model.py`, `src/models/price_encoder.py`, `src/data/graph_dataset.py`, and all `results/experiment[1-5]*` are read-only.
- Follow repo style: module docstring, type hints on public functions.

---

## Task 1: Correlation edge builder

**Files:**
- Create: `src/data/graph_edges.py`
- Test: `tests/test_graph_edges.py`

**Interfaces:**
- Produces: `load_returns_matrix(final_parquet_path: str) -> pd.DataFrame` (index=Timestamp date, columns=ticker, values=Return). `build_correlation_edges(returns: pd.DataFrame, target_date, window: int = 60, k: int = 5, min_periods: int = 10) -> set` returning a set of `(ticker_a, ticker_b)` tuples, each sorted alphabetically so `(a,b)` and `(b,a)` never both appear. Consumed by Tasks 3, 6.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph_edges.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.data.graph_edges import build_correlation_edges, load_returns_matrix


def _returns_frame():
    """5 tickers x 80 trading days. AAA/BBB move together, CCC/DDD move
    together, EEE is pure noise on its own schedule."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=80)
    base1 = rng.normal(0, 1, 80)
    base2 = rng.normal(0, 1, 80)
    return pd.DataFrame(
        {
            "AAA": base1 + rng.normal(0, 0.01, 80),
            "BBB": base1 + rng.normal(0, 0.01, 80),
            "CCC": base2 + rng.normal(0, 0.01, 80),
            "DDD": base2 + rng.normal(0, 0.01, 80),
            "EEE": rng.normal(0, 1, 80),
        },
        index=dates,
    )


def test_finds_the_correlated_pairs():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[70], window=60, k=1)
    assert ("AAA", "BBB") in edges
    assert ("CCC", "DDD") in edges


def test_edges_are_sorted_tuples_without_duplicates():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[70], window=60, k=2)
    for a, b in edges:
        assert a < b, "each edge must be a single alphabetically sorted tuple"
    assert len(edges) == len(set(edges))


def test_no_self_edges():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[70], window=60, k=3)
    assert all(a != b for a, b in edges)


def test_top_k_caps_degree():
    df = _returns_frame()
    k = 2
    edges = build_correlation_edges(df, df.index[70], window=60, k=k)
    # Each node proposes at most k partners; the union is undirected, so the
    # total can never exceed k * n_tickers (and is usually far less).
    assert len(edges) <= k * df.shape[1]


def test_leakage_target_day_cannot_change_edges():
    """THE leakage test: corrupting the target day and every day after it
    must not change the edges produced for that target day."""
    df = _returns_frame()
    target = df.index[70]
    clean = build_correlation_edges(df, target, window=60, k=2)

    poisoned = df.copy()
    poisoned.loc[target:] = 999.0   # target day onward is garbage
    after = build_correlation_edges(poisoned, target, window=60, k=2)

    assert clean == after, "edges for day t must not depend on data from t onward"


def test_insufficient_history_returns_empty_not_error():
    df = _returns_frame()
    edges = build_correlation_edges(df, df.index[2], window=60, k=5, min_periods=10)
    assert edges == set()


def test_min_periods_rejects_thin_overlap():
    """Two tickers overlapping on only a handful of days must not produce an
    edge, however perfectly those few days correlate."""
    dates = pd.bdate_range("2020-01-01", periods=80)
    df = pd.DataFrame(
        {
            "AAA": np.arange(80, dtype=float),
            "BBB": np.arange(80, dtype=float),
            "CCC": np.nan,
        },
        index=dates,
    )
    # CCC overlaps AAA on only 3 days, perfectly correlated on those days.
    df.iloc[60:63, df.columns.get_loc("CCC")] = [1.0, 2.0, 3.0]
    edges = build_correlation_edges(df, df.index[70], window=60, k=4, min_periods=10)
    assert ("AAA", "CCC") not in edges
    assert ("BBB", "CCC") not in edges


def test_load_returns_matrix_shape_on_real_data():
    df = load_returns_matrix("dataset/final/stocknet_final_modeling_set.parquet")
    assert df.shape == (504, 87)
    assert df.notna().sum().sum() == 26603
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_edges.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.graph_edges'`

- [ ] **Step 3: Write the implementation**

Create `src/data/graph_edges.py`:

```python
"""Leakage-safe edge builders for Experiment 6's dense stock graph.

Every builder takes an explicit time cutoff and returns undirected
`(ticker_a, ticker_b)` pairs for one target date, where each pair is a
single alphabetically sorted tuple. No builder may use data from the
target date or later -- see
docs/superpowers/specs/2026-09-03-dense-graph-experiment-design.md and the
leakage tests in tests/test_graph_edges.py.
"""

import pandas as pd


def load_returns_matrix(final_parquet_path: str) -> pd.DataFrame:
    """Loads a (date x ticker) matrix of daily returns from the corrected
    modeling set. The matrix is ~39% NaN: all 87 tickers have gaps inside
    their own active range, which is why correlation uses a 60-day window
    and a min_periods guard."""
    df = pd.read_parquet(final_parquet_path, columns=["Date", "Ticker", "Return"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.pivot_table(index="Date", columns="Ticker", values="Return").sort_index()


def build_correlation_edges(returns: pd.DataFrame, target_date, window: int = 60,
                            k: int = 5, min_periods: int = 10) -> set:
    """Connects each ticker to its top-`k` most correlated partners by |rho|,
    computed over the `window` trading days STRICTLY BEFORE `target_date`.

    Top-k rather than a |rho| threshold: a 0.5 threshold measured 2130 edges
    (57% density), which would rebuild the over-smoothing collapse diagnosed
    in Experiment 4. Top-k makes density a controlled parameter.

    Returns an empty set when there is insufficient history (early dates),
    rather than raising.
    """
    idx = returns.index.searchsorted(pd.Timestamp(target_date))
    win = returns.iloc[max(0, idx - window):idx]   # ends at t-1; excludes t
    if len(win) < min_periods:
        return set()

    corr = win.corr(min_periods=min_periods).abs()
    for ticker in corr.columns:
        corr.at[ticker, ticker] = float("nan")     # never self-edge

    edges = set()
    for ticker in corr.columns:
        partners = corr[ticker].dropna().sort_values(ascending=False)
        for partner in partners.head(k).index:
            edges.add(tuple(sorted((ticker, partner))))
    return edges
```

Note on the leakage guard: `searchsorted` returns the position of
`target_date` itself, and `iloc[:idx]` is exclusive of that position — so
the window ends at `t-1`. This is what
`test_leakage_target_day_cannot_change_edges` pins down.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_edges.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/graph_edges.py tests/test_graph_edges.py
git commit -m "Add leakage-safe correlation edge builder (Experiment 6)"
```

---

## Task 2: Co-mention edge builder

**Files:**
- Modify: `src/data/graph_edges.py` (append)
- Modify: `tests/test_graph_edges.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 beyond the module already existing.
- Produces: `load_event_texts(final_parquet_path: str) -> dict` mapping `pd.Timestamp -> list[str]` (all event-tweet texts on that date, across tickers). `extract_cashtags(text: str, universe: set) -> set` returning the universe tickers mentioned. `build_comention_edges(texts_by_date: dict, target_date, universe: set, lookback: int = 5) -> set` returning sorted `(a, b)` tuples. Consumed by Tasks 3, 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph_edges.py`:

```python
from src.data.graph_edges import (build_comention_edges, extract_cashtags,
                                  load_event_texts)

UNIVERSE = {"AAPL", "MSFT", "GOOG", "BAC"}


def test_extract_cashtags_handles_the_corpus_space_format():
    # The corpus writes cashtags as "$ aapl", with a space after the dollar.
    got = extract_cashtags("$ aapl - online retail sales up -> URL $ msft", UNIVERSE)
    assert got == {"AAPL", "MSFT"}


def test_extract_cashtags_ignores_tickers_outside_the_universe():
    got = extract_cashtags("$ aapl and $ tsla and $ zzzz", UNIVERSE)
    assert got == {"AAPL"}


def test_comention_pairs_every_mentioned_ticker():
    texts = {pd.Timestamp("2020-01-06"): ["$ aapl beats $ msft and $ goog today"]}
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-07"), UNIVERSE, lookback=5)
    assert edges == {("AAPL", "GOOG"), ("AAPL", "MSFT"), ("GOOG", "MSFT")}


def test_single_ticker_mention_creates_no_edge():
    texts = {pd.Timestamp("2020-01-06"): ["$ aapl going up"]}
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-07"), UNIVERSE, lookback=5)
    assert edges == set()


def test_leakage_target_day_tweets_are_excluded():
    """Tweets dated on the target day itself must never inform that day."""
    texts = {
        pd.Timestamp("2020-01-06"): ["$ aapl with $ msft"],
        pd.Timestamp("2020-01-07"): ["$ goog with $ bac"],   # the target day
    }
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-07"), UNIVERSE, lookback=5)
    assert ("AAPL", "MSFT") in edges
    assert ("BAC", "GOOG") not in edges, "target-day tweets must be excluded"


def test_lookback_window_is_respected():
    texts = {
        pd.Timestamp("2020-01-01"): ["$ aapl with $ msft"],   # far in the past
        pd.Timestamp("2020-01-20"): ["$ goog with $ bac"],    # just before target
    }
    edges = build_comention_edges(texts, pd.Timestamp("2020-01-21"), UNIVERSE, lookback=3)
    assert ("BAC", "GOOG") in edges
    assert ("AAPL", "MSFT") not in edges, "outside the lookback window"


def test_load_event_texts_on_real_data():
    texts = load_event_texts("dataset/final/stocknet_final_modeling_set.parquet")
    assert len(texts) > 300
    assert all(isinstance(v, list) for v in texts.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_edges.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_comention_edges'`

- [ ] **Step 3: Write the implementation**

Append to `src/data/graph_edges.py`:

```python
import re

_CASHTAG = re.compile(r"\$\s*([A-Za-z]{1,5})")


def extract_cashtags(text: str, universe: set) -> set:
    """Pulls universe tickers out of a tweet. The corpus writes cashtags with
    a space after the dollar sign (`$ aapl`), hence the `\\s*`."""
    found = {m.upper() for m in _CASHTAG.findall(str(text))}
    return found & universe


def load_event_texts(final_parquet_path: str) -> dict:
    """Maps each date to the list of that date's event-tweet texts, pooled
    across tickers. Rows whose Event_Texts is empty are skipped."""
    df = pd.read_parquet(final_parquet_path, columns=["Date", "Event_Texts"])
    df["Date"] = pd.to_datetime(df["Date"])
    out: dict = {}
    for date, text in zip(df["Date"], df["Event_Texts"]):
        if text is None or len(str(text)) <= 2:
            continue
        out.setdefault(date, []).extend(str(text).split("[SEP]"))
    return out


def build_comention_edges(texts_by_date: dict, target_date, universe: set,
                          lookback: int = 5) -> set:
    """Connects tickers that appear together in the same tweet, over the
    `lookback` days STRICTLY BEFORE `target_date`.

    Measured on the real corpus: 1785 tweet segments mention 2+ universe
    tickers, spread across 347 of 504 trading days.
    """
    target = pd.Timestamp(target_date)
    window_start = target - pd.Timedelta(days=lookback)

    edges = set()
    for date, texts in texts_by_date.items():
        if not (window_start <= date < target):    # excludes the target day
            continue
        for text in texts:
            mentioned = sorted(extract_cashtags(text, universe))
            for i in range(len(mentioned)):
                for j in range(i + 1, len(mentioned)):
                    edges.add((mentioned[i], mentioned[j]))
    return edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_edges.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/graph_edges.py tests/test_graph_edges.py
git commit -m "Add leakage-safe tweet co-mention edge builder (Experiment 6)"
```

---

## Task 3: Edge combiner producing edge_index + edge_type

**Files:**
- Modify: `src/data/graph_edges.py` (append)
- Modify: `tests/test_graph_edges.py` (append)

**Interfaces:**
- Consumes: `build_correlation_edges`, `build_comention_edges` from Tasks 1-2.
- Produces: `build_sector_edges(sector_csv_path: str) -> set` (sorted tuples, same 381 edges as Experiment 4's graph). `combine_edges(edge_sets: dict, tickers: list) -> tuple[torch.Tensor, torch.Tensor]` returning `(edge_index (2,E), edge_type (E,))` with local indices into `tickers`, both directions per undirected edge, relation ids `0=sector, 1=correlation, 2=comention`. Consumed by Tasks 4, 5, 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph_edges.py`:

```python
import torch

from src.data.graph_edges import build_sector_edges, combine_edges

RELATION = {"sector": 0, "correlation": 1, "comention": 2}


def test_build_sector_edges_matches_experiment_4_graph():
    edges = build_sector_edges("dataset/final/sector_mapping.csv")
    assert len(edges) == 381
    assert all(a < b for a, b in edges)


def test_combine_edges_uses_local_indices_and_both_directions():
    tickers = ["AAA", "BBB", "CCC"]
    ei, et = combine_edges({"sector": {("AAA", "BBB")}}, tickers)
    pairs = set(map(tuple, ei.t().tolist()))
    assert pairs == {(0, 1), (1, 0)}
    assert et.tolist() == [0, 0]


def test_combine_edges_tags_each_relation():
    tickers = ["AAA", "BBB", "CCC"]
    ei, et = combine_edges(
        {"sector": {("AAA", "BBB")}, "comention": {("BBB", "CCC")}}, tickers
    )
    by_pair = {tuple(p): t for p, t in zip(ei.t().tolist(), et.tolist())}
    assert by_pair[(0, 1)] == RELATION["sector"]
    assert by_pair[(1, 2)] == RELATION["comention"]


def test_overlapping_pair_takes_lowest_relation_id():
    """A pair present in two relations appears once, tagged with the
    lowest-numbered relation (documented tie-break)."""
    tickers = ["AAA", "BBB"]
    ei, et = combine_edges(
        {"correlation": {("AAA", "BBB")}, "sector": {("AAA", "BBB")}}, tickers
    )
    assert ei.shape == (2, 2)                 # one undirected edge, both directions
    assert set(et.tolist()) == {RELATION["sector"]}


def test_combine_edges_drops_tickers_not_in_the_snapshot():
    tickers = ["AAA", "BBB"]
    ei, et = combine_edges({"sector": {("AAA", "ZZZ")}}, tickers)
    assert ei.shape == (2, 0)
    assert et.shape == (0,)


def test_combine_edges_empty_input_is_well_formed():
    ei, et = combine_edges({}, ["AAA", "BBB"])
    assert ei.shape == (2, 0) and ei.dtype == torch.long
    assert et.shape == (0,) and et.dtype == torch.long
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_edges.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_sector_edges'`

- [ ] **Step 3: Write the implementation**

Append to `src/data/graph_edges.py`:

```python
from itertools import combinations

import torch

RELATION_IDS = {"sector": 0, "correlation": 1, "comention": 2}


def build_sector_edges(sector_csv_path: str) -> set:
    """Static edges: two tickers connect iff they share a Sector. Produces
    the same 381 edges as Experiment 4's graph (9 disjoint complete
    cliques), so sector-only cells of this grid re-run that experiment."""
    df = pd.read_csv(sector_csv_path)
    edges = set()
    for _, group in df.groupby("Sector"):
        for a, b in combinations(sorted(group["Ticker"]), 2):
            edges.add((a, b))
    return edges


def combine_edges(edge_sets: dict, tickers: list):
    """Merges named edge sets into a (2,E) edge_index plus an (E,) edge_type,
    using each ticker's position in `tickers` as its node index.

    Pairs whose endpoints are not both in `tickers` are dropped. A pair
    present in several relations is emitted once, tagged with the
    lowest-numbered relation (sector < correlation < comention).
    """
    index_of = {t: i for i, t in enumerate(tickers)}

    relation_of_pair: dict = {}
    for name, pairs in edge_sets.items():
        rel = RELATION_IDS[name]
        for a, b in pairs:
            if a not in index_of or b not in index_of:
                continue
            key = (a, b) if a < b else (b, a)
            if rel < relation_of_pair.get(key, 99):
                relation_of_pair[key] = rel

    if not relation_of_pair:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    src, dst, types = [], [], []
    for (a, b), rel in relation_of_pair.items():
        ia, ib = index_of[a], index_of[b]
        src.extend([ia, ib])
        dst.extend([ib, ia])
        types.extend([rel, rel])
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(types, dtype=torch.long))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_edges.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/graph_edges.py tests/test_graph_edges.py
git commit -m "Add sector edges and multi-relation edge combiner (Experiment 6)"
```

---

## Task 4: GraphSAGE and R-GCN models

**Files:**
- Create: `src/models/graph_models.py`
- Test: `tests/test_graph_models.py`

**Interfaces:**
- Consumes: `PriceEncoder(input_dim: int = 3, hidden_dim: int = 64)` with `.forward(x, mask) -> (N, hidden_dim)` from `src.models.price_encoder`.
- Produces: `StockGraphSAGE(price_input_dim=3, hidden_dim=64, dropout=0.2)` and `StockRGCN(price_input_dim=3, hidden_dim=64, num_relations=3, dropout=0.2)`, both with `.forward(price_seq, price_mask, edge_index, edge_type=None) -> (N, 2)`. Consumed by Task 6.

Both models take `edge_type` in their signature (ignored by GraphSAGE) so the grid runner can call every model identically.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph_models.py`:

```python
from itertools import combinations

import torch

from src.models.graph_models import StockGraphSAGE, StockRGCN


def _clique(n):
    src, dst = [], []
    for a, b in combinations(range(n), 2):
        src.extend([a, b])
        dst.extend([b, a])
    return torch.tensor([src, dst], dtype=torch.long)


def test_graphsage_forward_shape():
    model = StockGraphSAGE(hidden_dim=16)
    logits = model(torch.randn(5, 4, 3), torch.ones(5, 4), _clique(5))
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_rgcn_forward_shape():
    model = StockRGCN(hidden_dim=16)
    ei = _clique(5)
    et = torch.zeros(ei.shape[1], dtype=torch.long)
    logits = model(torch.randn(5, 4, 3), torch.ones(5, 4), ei, et)
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_rgcn_handles_all_three_relations():
    model = StockRGCN(hidden_dim=16, num_relations=3)
    ei = _clique(6)
    et = torch.randint(0, 3, (ei.shape[1],))
    logits = model(torch.randn(6, 4, 3), torch.ones(6, 4), ei, et)
    assert logits.shape == (6, 2)
    assert torch.isfinite(logits).all()


def test_models_handle_empty_edge_index():
    """Early dates can have no edges at all; models must still run."""
    ei = torch.empty((2, 0), dtype=torch.long)
    et = torch.empty((0,), dtype=torch.long)
    for logits in (
        StockGraphSAGE(hidden_dim=16)(torch.randn(3, 4, 3), torch.ones(3, 4), ei),
        StockRGCN(hidden_dim=16)(torch.randn(3, 4, 3), torch.ones(3, 4), ei, et),
    ):
        assert logits.shape == (3, 2)
        assert torch.isfinite(logits).all()


def test_graphsage_does_not_collapse_on_a_clique_after_training():
    """Experiment 4's over-smoothing regression, ported: on a complete
    clique, node outputs must stay distinguishable after training."""
    torch.manual_seed(0)
    n = 8
    model = StockGraphSAGE(hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, mask, ei = torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(x, mask, ei), target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x, mask, ei)
    assert logits.std(dim=0).mean().item() > 1e-4


def test_rgcn_does_not_collapse_on_a_clique_after_training():
    torch.manual_seed(0)
    n = 8
    model = StockRGCN(hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, mask, ei = torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n)
    et = torch.zeros(ei.shape[1], dtype=torch.long)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(x, mask, ei, et), target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x, mask, ei, et)
    assert logits.std(dim=0).mean().item() > 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.graph_models'`

- [ ] **Step 3: Write the implementation**

Create `src/models/graph_models.py`:

```python
"""GraphSAGE and R-GCN variants for Experiment 6's dense stock graph.

Both wrap each message-passing layer in a residual connection, for the same
reason StockGAT does (see results/experiment4_gat/EXPERIMENT4_GAT.md): the
sector graph is a set of complete cliques, and plain stacked message
passing over a complete graph collapses every node to an identical output.

R-GCN is the only model here that learns a separate transformation per
edge type, which is why it is included -- it is the natural fit for a graph
whose edges mean three different things.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, SAGEConv

from src.models.price_encoder import PriceEncoder


class StockGraphSAGE(nn.Module):
    """PriceEncoder per ticker -> 2 residual SAGEConv layers -> classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        h = self.price_encoder(price_seq, price_mask)
        h = h + torch.relu(self.conv1(h, edge_index))
        h = self.dropout(h)
        h = h + torch.relu(self.conv2(h, edge_index))
        return self.classifier(h)


class StockRGCN(nn.Module):
    """PriceEncoder per ticker -> 2 residual RGCNConv layers -> classifier.

    `num_relations` is always 3 even when the active edge combination uses
    fewer; the unused relation's weights simply receive no gradient. This
    keeps one model definition across all 7 edge combinations.
    """

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 num_relations: int = 3, dropout: float = 0.2):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        h = self.price_encoder(price_seq, price_mask)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long,
                                    device=edge_index.device)
        h = h + torch.relu(self.conv1(h, edge_index, edge_type))
        h = self.dropout(h)
        h = h + torch.relu(self.conv2(h, edge_index, edge_type))
        return self.classifier(h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_models.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/graph_models.py tests/test_graph_models.py
git commit -m "Add StockGraphSAGE and StockRGCN models (Experiment 6)"
```

---

## Task 5: Multi-relation bias for the graph transformer

**Files:**
- Create: `src/models/graph_transformer_multi.py`
- Test: `tests/test_graph_transformer_multi.py`

**Interfaces:**
- Consumes: `PriceEncoder` from `src.models.price_encoder`.
- Produces: `MultiRelationGraphTransformer(price_input_dim=3, hidden_dim=64, nhead=4, dim_feedforward=128, dropout=0.2, num_relations=3)` with `.forward(price_seq, price_mask, edge_index, edge_type=None) -> (N, 2)`. Consumed by Task 6.

This is a **new file**, not an edit to `src/models/graph_transformer.py` — Experiment 5's model and results must stay reproducible exactly as published. The new model generalises Experiment 5's single learned scalar bias to one learned scalar per relation, and takes `edge_index`/`edge_type` so the grid runner calls every model with one signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph_transformer_multi.py`:

```python
import torch

from src.models.graph_transformer_multi import MultiRelationGraphTransformer


def test_forward_shape():
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    et = torch.tensor([0, 0, 1, 1])
    logits = model(torch.randn(5, 4, 3), torch.ones(5, 4), ei, et)
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_empty_edge_index_still_runs():
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    ei = torch.empty((2, 0), dtype=torch.long)
    et = torch.empty((0,), dtype=torch.long)
    logits = model(torch.randn(3, 4, 3), torch.ones(3, 4), ei, et)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_each_relation_has_its_own_learnable_bias():
    model = MultiRelationGraphTransformer(num_relations=3)
    assert model.relation_bias.shape == (3,)
    assert model.relation_bias.requires_grad


def test_relation_bias_changes_the_output():
    """A non-zero bias on a relation must actually alter attention."""
    torch.manual_seed(0)
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    model.eval()
    ei = torch.tensor([[0, 1], [1, 0]])
    et = torch.tensor([0, 0])
    x, mask = torch.randn(4, 4, 3), torch.ones(4, 4)
    with torch.no_grad():
        model.relation_bias.zero_()
        flat = model(x, mask, ei, et)
        model.relation_bias[0] = 5.0
        biased = model(x, mask, ei, et)
    assert not torch.allclose(flat, biased)


def test_relations_are_independent():
    """Changing relation 1's bias must not move a graph that only uses
    relation 0."""
    torch.manual_seed(0)
    model = MultiRelationGraphTransformer(hidden_dim=16, nhead=2, dim_feedforward=32)
    model.eval()
    ei = torch.tensor([[0, 1], [1, 0]])
    et = torch.tensor([0, 0])
    x, mask = torch.randn(4, 4, 3), torch.ones(4, 4)
    with torch.no_grad():
        model.relation_bias.zero_()
        before = model(x, mask, ei, et)
        model.relation_bias[1] = 5.0        # a relation not present in et
        after = model(x, mask, ei, et)
    assert torch.allclose(before, after)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_transformer_multi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.graph_transformer_multi'`

- [ ] **Step 3: Write the implementation**

Create `src/models/graph_transformer_multi.py`:

```python
"""Graph transformer with one learned attention bias per edge relation.

Generalises Experiment 5's StockGraphTransformer (which had a single
learned scalar for "same sector") to a graph whose edges mean three
different things. Experiment 5's model is left untouched in
src/models/graph_transformer.py so its published results stay reproducible.

Attention is global -- every ticker attends to every other ticker in the
day's cross-section -- and the graph enters as an additive bias on the raw
attention scores, the Graphormer-style mechanism nn.TransformerEncoderLayer
supports natively via a float src_mask.
"""

import torch
import torch.nn as nn

from src.models.price_encoder import PriceEncoder


class MultiRelationGraphTransformer(nn.Module):
    """PriceEncoder -> 2 global-attention layers biased per relation ->
    per-ticker classifier."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 nhead: int = 4, dim_feedforward: int = 128,
                 dropout: float = 0.2, num_relations: int = 3):
        super().__init__()
        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.relation_bias = nn.Parameter(torch.zeros(num_relations))
        layer_kwargs = dict(d_model=hidden_dim, nhead=nhead,
                            dim_feedforward=dim_feedforward, dropout=dropout,
                            batch_first=True)
        self.layer1 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.layer2 = nn.TransformerEncoderLayer(**layer_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )

    def _bias_matrix(self, n: int, edge_index: torch.Tensor,
                     edge_type: torch.Tensor) -> torch.Tensor:
        """(N,N) additive attention bias: entry [i,j] is the learned scalar
        for the relation connecting i and j, or 0 where they are unconnected."""
        bias = torch.zeros(n, n, device=self.relation_bias.device)
        if edge_index.numel() == 0:
            return bias
        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long,
                                    device=edge_index.device)
        bias[edge_index[0], edge_index[1]] = self.relation_bias[edge_type]
        return bias

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        h = self.price_encoder(price_seq, price_mask)
        bias = self._bias_matrix(h.shape[0], edge_index, edge_type)
        x = h.unsqueeze(0)                       # (1, N, hidden) for batch_first
        x = self.layer1(x, src_mask=bias)
        x = self.layer2(x, src_mask=bias)
        return self.classifier(x.squeeze(0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_transformer_multi.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/graph_transformer_multi.py tests/test_graph_transformer_multi.py
git commit -m "Add multi-relation graph transformer (Experiment 6)"
```

---

## Task 6: Per-day edge cache builder

**Files:**
- Create: `src/data/edge_cache.py`
- Test: `tests/test_edge_cache.py`

**Interfaces:**
- Consumes: `load_returns_matrix`, `load_event_texts`, `build_sector_edges`, `build_correlation_edges`, `build_comention_edges` from Tasks 1-3.
- Produces: `build_edge_cache(exact_parquet, final_parquet, sector_csv, out_path, window=60, k=5, lookback=5) -> dict` mapping `date_string -> {"sector": set, "correlation": set, "comention": set}`, pickled to `out_path`. `load_edge_cache(path) -> dict`. Consumed by Task 7.

Computing correlation edges is the slowest part of the pipeline (a 87x87 correlation per trading day). Caching it once means the 84-run grid pays that cost a single time, and a crashed Colab session reloads instead of recomputing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_edge_cache.py`:

```python
import pickle

from src.data.edge_cache import build_edge_cache, load_edge_cache


def test_builds_and_roundtrips_a_cache(tmp_path):
    out = tmp_path / "edges.pkl"
    cache = build_edge_cache(
        "dataset/stocknet_protocol_a_exact.parquet",
        "dataset/final/stocknet_final_modeling_set.parquet",
        "dataset/final/sector_mapping.csv",
        str(out),
    )
    assert out.exists()
    assert len(cache) == 504, "one entry per trading day in the exact-protocol set"

    reloaded = load_edge_cache(str(out))
    assert reloaded.keys() == cache.keys()

    sample = cache[sorted(cache)[300]]
    assert set(sample) == {"sector", "correlation", "comention"}
    assert len(sample["sector"]) == 381
    assert len(sample["correlation"]) > 0


def test_early_dates_have_empty_correlation_but_full_sector(tmp_path):
    """Warm-up behaviour: the first trading days lack the history for a
    60-day correlation window, and must degrade to empty, not crash."""
    cache = build_edge_cache(
        "dataset/stocknet_protocol_a_exact.parquet",
        "dataset/final/stocknet_final_modeling_set.parquet",
        "dataset/final/sector_mapping.csv",
        str(tmp_path / "edges.pkl"),
    )
    first = cache[sorted(cache)[0]]
    assert first["correlation"] == set()
    assert len(first["sector"]) == 381
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_edge_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.edge_cache'`

- [ ] **Step 3: Write the implementation**

Create `src/data/edge_cache.py`:

```python
"""Precomputes per-trading-day edge sets for Experiment 6 and caches them.

The correlation builder runs an 87x87 correlation per trading day; without
this cache the 84-run grid would repeat that work 84 times. The cache is
also what lets a crashed Colab session resume rather than restart.
"""

import pickle

import pandas as pd

from src.data.graph_edges import (build_comention_edges, build_correlation_edges,
                                  build_sector_edges, load_event_texts,
                                  load_returns_matrix)


def build_edge_cache(exact_parquet: str, final_parquet: str, sector_csv: str,
                     out_path: str, window: int = 60, k: int = 5,
                     lookback: int = 5) -> dict:
    """Builds {date_string: {relation: set_of_pairs}} for every trading day in
    the exact-protocol dataset, and pickles it to `out_path`."""
    exact = pd.read_parquet(exact_parquet, columns=["Target_Date"])
    dates = sorted(pd.to_datetime(exact["Target_Date"]).unique())

    returns = load_returns_matrix(final_parquet)
    texts = load_event_texts(final_parquet)
    universe = set(returns.columns)
    sector = build_sector_edges(sector_csv)

    cache = {}
    for date in dates:
        ts = pd.Timestamp(date)
        cache[ts.strftime("%Y-%m-%d")] = {
            "sector": sector,
            "correlation": build_correlation_edges(returns, ts, window=window, k=k),
            "comention": build_comention_edges(texts, ts, universe, lookback=lookback),
        }

    with open(out_path, "wb") as fh:
        pickle.dump(cache, fh)
    return cache


def load_edge_cache(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_edge_cache.py -v`
Expected: 2 passed (may take ~1-2 minutes; it computes 504 correlation matrices)

- [ ] **Step 5: Commit**

```bash
git add src/data/edge_cache.py tests/test_edge_cache.py
git commit -m "Add per-day edge cache builder (Experiment 6)"
```

---

## Task 7: Crash-resilient grid runner

**Files:**
- Create: `scripts/run_dense_graph_grid.py`

**Interfaces:**
- Consumes: `build_edge_cache`/`load_edge_cache` (Task 6), `combine_edges`/`RELATION_IDS` (Task 3), `StockGraphSAGE`/`StockRGCN` (Task 4), `MultiRelationGraphTransformer` (Task 5), `StockGAT` from `src.models.graph_model`, `build_daily_snapshots`/`normalize_price_snapshots` from `src.data.graph_dataset`, `set_seed` from `src.utils.seed`.
- Produces: `results/experiment6_dense_graph/results.csv` (one row per completed run) and one `metrics.json` per run.

The training loop reuses Experiment 4's fixed pattern verbatim — gradient accumulation over 16 days, shuffled day order, early stopping on validation MCC — because that fix is what made graph training work at all (see `results/experiment4_gat/EXPERIMENT4_GAT.md`).

- [ ] **Step 1: Write the runner**

Create `scripts/run_dense_graph_grid.py`:

```python
"""Runs Experiment 6's escalating grid: every non-empty combination of
{sector, correlation, comention} edges, across four graph models and
multiple seeds.

Results are appended after EVERY run and completed cells are skipped on
restart, so a crashed Colab session resumes where it stopped. Pass --pass 1,
2, or 3 to control scope; each pass is a superset of the previous one and
re-runs nothing.
"""

import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.data.edge_cache import build_edge_cache, load_edge_cache
from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots
from src.data.graph_edges import combine_edges
from src.models.graph_model import StockGAT
from src.models.graph_models import StockGraphSAGE, StockRGCN
from src.models.graph_transformer_multi import MultiRelationGraphTransformer
from src.utils.seed import set_seed

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
OUT_DIR = "results/experiment6_dense_graph"
EDGE_CACHE = os.path.join(OUT_DIR, "edge_cache.pkl")
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")

PATIENCE, MAX_EPOCHS, LR, GRAD_ACCUM_DAYS = 10, 100, 1e-3, 16
RELATIONS = ["sector", "correlation", "comention"]


class GATAdapter(nn.Module):
    """Experiment 4's StockGAT.forward takes (price_seq, price_mask,
    edge_index) with no edge_type. This adapter gives it the same 4-argument
    signature as every other model so the grid runner can call them
    identically, without editing StockGAT (whose published Experiment 4
    results must stay reproducible)."""

    def __init__(self):
        super().__init__()
        self.inner = StockGAT()

    def forward(self, price_seq, price_mask, edge_index, edge_type=None):
        return self.inner(price_seq, price_mask, edge_index)


MODELS = {
    "GAT": lambda: GATAdapter(),
    "GraphTransformer": lambda: MultiRelationGraphTransformer(),
    "GraphSAGE": lambda: StockGraphSAGE(),
    "RGCN": lambda: StockRGCN(),
}
PASS_SPEC = {
    1: (["GAT", "GraphTransformer"], [0]),
    2: (["GAT", "GraphTransformer"], [0, 1, 2]),
    3: (list(MODELS), [0, 1, 2]),
}


def edge_combos():
    """All 7 non-empty subsets of the three relations, shortest first."""
    out = []
    for r in (1, 2, 3):
        for combo in itertools.combinations(RELATIONS, r):
            out.append(list(combo))
    return out


def run_epoch(model, snapshots, cache, combo, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        snapshots = list(snapshots)
        random.shuffle(snapshots)
        optimizer.zero_grad()

    probs_all, targets_all, total_loss = [], [], 0.0
    for i, snap in enumerate(snapshots):
        day = cache.get(snap.date, {})
        edge_sets = {r: day.get(r, set()) for r in combo}
        edge_index, edge_type = combine_edges(edge_sets, snap.tickers)
        logits = model(snap.price_seq, snap.price_mask, edge_index, edge_type)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            (loss / GRAD_ACCUM_DAYS).backward()
            if (i + 1) % GRAD_ACCUM_DAYS == 0 or i == len(snapshots) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * len(snap.tickers)
        probs_all.append(torch.softmax(logits, dim=-1)[:, 1].detach())
        targets_all.append(snap.target)

    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy()
    preds = (probs >= 0.5).astype(int)
    return total_loss / len(targets), {
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "auc": roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.5,
    }


def train_one(model_name, combo, seed, snaps, cache):
    set_seed(seed)
    train_snaps, dev_snaps, test_snaps = snaps
    model = MODELS[model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_mcc, best_epoch, best_state, stale = -1.0, -1, None, 0
    for epoch in range(MAX_EPOCHS):
        _, _ = run_epoch(model, train_snaps, cache, combo, optimizer)
        _, val = run_epoch(model, dev_snaps, cache, combo)
        if val["mcc"] > best_mcc:
            best_mcc, best_epoch, stale = val["mcc"], epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    _, test = run_epoch(model, test_snaps, cache, combo)
    test["n_samples"] = sum(len(s.tickers) for s in test_snaps)
    test["best_epoch"] = best_epoch
    test["best_val_mcc"] = best_mcc
    return test


def already_done(model_name, combo, seed):
    if not os.path.exists(RESULTS_CSV):
        return False
    df = pd.read_csv(RESULTS_CSV)
    key = (df["model"] == model_name) & (df["edges"] == "+".join(combo)) & (df["seed"] == seed)
    return bool(key.any())


def record(model_name, combo, seed, test):
    label = f"{model_name} [{'+'.join(combo)}] seed={seed}"
    run_dir = os.path.join(OUT_DIR, f"{model_name}__{'_'.join(combo)}__seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump({"test": {"label": label, "backend": None, **test}}, fh, indent=2)

    row = {"model": model_name, "edges": "+".join(combo), "seed": seed, **test}
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_num", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(EDGE_CACHE):
        cache = load_edge_cache(EDGE_CACHE)
        print(f"Loaded edge cache ({len(cache)} days)")
    else:
        print("Building edge cache (one-time, ~1-2 min)...")
        cache = build_edge_cache(EXACT, FINAL, SECTOR_CSV, EDGE_CACHE)

    df = pd.read_parquet(EXACT)
    train_snaps = build_daily_snapshots(df, "train")
    dev_snaps = build_daily_snapshots(df, "dev")
    test_snaps = build_daily_snapshots(df, "test")
    normalize_price_snapshots(train_snaps, dev_snaps, test_snaps)
    snaps = (train_snaps, dev_snaps, test_snaps)

    model_names, seeds = PASS_SPEC[args.pass_num]
    todo = [(m, c, s) for m in model_names for c in edge_combos() for s in seeds]
    print(f"Pass {args.pass_num}: {len(todo)} cells in grid")

    for model_name, combo, seed in todo:
        if already_done(model_name, combo, seed):
            print(f"  skip  {model_name} [{'+'.join(combo)}] seed={seed}")
            continue
        print(f"  train {model_name} [{'+'.join(combo)}] seed={seed} ...", flush=True)
        test = train_one(model_name, combo, seed, snaps, cache)
        record(model_name, combo, seed, test)
        print(f"        MCC={test['mcc']:+.4f} F1={test['f1']:.4f} AUC={test['auc']:.4f}")

    print(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `date` to DaySnapshot**

The runner needs each snapshot to know its own date so it can look up that
day's edges. `DaySnapshot` currently has no date field.

Modify `src/data/graph_dataset.py` — add a `date: str = ""` field to the
dataclass and populate it in `build_daily_snapshots`:

```python
@dataclass
class DaySnapshot:
    tickers: list
    price_seq: torch.Tensor   # (N, 4, 3)
    price_mask: torch.Tensor  # (N, 4)
    target: torch.Tensor      # (N,)
    date: str = ""            # "YYYY-MM-DD"; used by Experiment 6 to look up that day's edges
```

and in `build_daily_snapshots`'s loop, pass `date=str(day_value)[:10]` when
constructing each `DaySnapshot`, where `day_value` is the `Target_Date`
group key.

The default `""` keeps every existing call site and Experiment 4/5 test
working unchanged.

- [ ] **Step 3: Verify existing tests still pass**

Run: `pytest tests/ -v`
Expected: all previously passing tests still pass (the dataclass change is
backwards-compatible via the default).

- [ ] **Step 4: Smoke-test the runner on one cell**

Run:
```bash
python -c "
import sys; sys.path.insert(0,'.')
from scripts.run_dense_graph_grid import edge_combos
print(edge_combos())
assert len(edge_combos()) == 7
print('OK: 7 edge combinations')
"
```
Expected: prints the 7 combinations and `OK: 7 edge combinations`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_dense_graph_grid.py src/data/graph_dataset.py
git commit -m "Add crash-resilient dense-graph grid runner (Experiment 6)"
```

---

## Task 8: Figures

**Files:**
- Create: `scripts/make_experiment6_figures.py`

**Interfaces:**
- Consumes: `load_edge_cache` (Task 6), `results/experiment6_dense_graph/results.csv` (Task 7).
- Produces: four PNGs in `results/experiment6_dense_graph/figures/`.

- [ ] **Step 1: Write the figures script**

Create `scripts/make_experiment6_figures.py`:

```python
"""Generates Experiment 6's figures from the edge cache and results table."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.data.edge_cache import load_edge_cache

OUT_DIR = "results/experiment6_dense_graph"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SECTOR_CSV = "dataset/final/sector_mapping.csv"
BASELINES = {"Exp1 LSTM+tweets": 0.0922, "Exp3 price-only": 0.0108, "Exp4 sector GAT": 0.0278}


def draw_graph(pairs, sectors, title, path):
    g = nx.Graph()
    g.add_nodes_from(sectors)
    g.add_edges_from(pairs)
    codes = sorted(set(sectors.values()))
    colors = [codes.index(sectors[n]) for n in g.nodes()]
    plt.figure(figsize=(9, 9))
    nx.draw_networkx(
        g, pos=nx.spring_layout(g, seed=42, k=0.35), node_color=colors,
        cmap="tab10", node_size=220, font_size=6, edge_color="#bbbbbb", width=0.5,
    )
    plt.title(f"{title}\n{g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sectors = pd.read_csv(SECTOR_CSV).set_index("Ticker")["Sector"].to_dict()
    cache = load_edge_cache(os.path.join(OUT_DIR, "edge_cache.pkl"))
    day = cache[sorted(cache)[300]]     # a mid-dataset day, past correlation warm-up

    for rel, title in [("sector", "Sector edges"), ("correlation", "Correlation edges (60d, top-5)"),
                       ("comention", "Tweet co-mention edges")]:
        draw_graph(day[rel], sectors, title, os.path.join(FIG_DIR, f"graph_{rel}.png"))

    # Edge density over time
    dates = sorted(cache)
    plt.figure(figsize=(12, 4))
    for rel in ("sector", "correlation", "comention"):
        plt.plot(range(len(dates)), [len(cache[d][rel]) for d in dates], label=rel, linewidth=1)
    plt.xlabel("trading day index"); plt.ylabel("edges")
    plt.title("Edge count per trading day, by relation")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "edge_density_over_time.png"), dpi=150)
    plt.close()

    results_csv = os.path.join(OUT_DIR, "results.csv")
    if not os.path.exists(results_csv):
        print("No results.csv yet; skipping result figures.")
        return
    df = pd.read_csv(results_csv)

    # Mean MCC per (model, edge combo), with seed spread as error bars
    agg = df.groupby(["model", "edges"])["mcc"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(13, 6))
    models = sorted(agg["model"].unique())
    combos = sorted(agg["edges"].unique())
    width = 0.8 / max(len(models), 1)
    for mi, m in enumerate(models):
        sub = agg[agg["model"] == m].set_index("edges").reindex(combos)
        xs = [i + mi * width for i in range(len(combos))]
        plt.bar(xs, sub["mean"].fillna(0), width=width,
                yerr=sub["std"].fillna(0), capsize=2, label=m)
    for name, val in BASELINES.items():
        plt.axhline(val, linestyle="--", linewidth=1, alpha=0.7)
        plt.text(len(combos) - 0.5, val, f" {name} ({val:+.4f})", va="bottom", fontsize=7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks([i + 0.4 for i in range(len(combos))], combos, rotation=20, ha="right", fontsize=8)
    plt.ylabel("test MCC (mean over seeds)")
    plt.title("Experiment 6: test MCC by model and edge combination")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mcc_by_model_and_edges.png"), dpi=150)
    plt.close()

    # Seed spread, so single-seed noise is visible rather than implied
    if df["seed"].nunique() > 1:
        plt.figure(figsize=(13, 5))
        labels, spreads = [], []
        for (m, e), grp in df.groupby(["model", "edges"]):
            labels.append(f"{m}\n{e}")
            spreads.append(grp["mcc"].max() - grp["mcc"].min())
        plt.bar(range(len(labels)), spreads)
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.ylabel("max - min test MCC across seeds")
        plt.title("Seed spread per grid cell (differences smaller than this are noise)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "seed_variance.png"), dpi=150)
        plt.close()

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs**

Run: `python scripts/make_experiment6_figures.py`
Expected: builds/loads the edge cache, writes `graph_sector.png`,
`graph_correlation.png`, `graph_comention.png`, and
`edge_density_over_time.png`; prints the "No results.csv yet" line if the
grid has not run.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_experiment6_figures.py
git commit -m "Add Experiment 6 figure generation"
```

---

## Task 9: Colab notebook

**Files:**
- Create: `notebooks/MMGTFFF_dense_graph_experiment.ipynb`

**Interfaces:**
- Consumes: everything above, via a repo clone.
- Produces: nothing the repo imports; it is the user-facing runner.

The notebook is deliberately thin — all logic lives in tested repo code, so
the notebook only clones, installs, runs, and displays.

- [ ] **Step 1: Create the notebook**

Create `notebooks/MMGTFFF_dense_graph_experiment.ipynb` with these cells,
in order. Build it with `nbformat` (a small Python script that writes the
JSON) rather than hand-writing notebook JSON.

Cell 1 (markdown):
```
# Experiment 6 — Dense Stock Graph + Model Grid

Runs every non-empty combination of {sector, correlation, co-mention} edges
across four graph models and three seeds, then adds text encoders.

Runtime -> Change runtime type -> **GPU** before running.

Results are written after every single run and completed cells are skipped
on restart, so if this session crashes, just re-run from the top.
```

Cell 2 (code) — clone and install:
```python
!git clone https://github.com/AdityaMelkote3004/capstone.git repo 2>/dev/null || echo "already cloned"
%cd repo
!git pull
!pip install -q torch_geometric
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
```

Cell 3 (code) — mount Drive so results survive a crashed runtime:
```python
from google.colab import drive
drive.mount('/content/drive')
import os
PERSIST = '/content/drive/MyDrive/mmgtfff_experiment6'
os.makedirs(PERSIST, exist_ok=True)
# Symlink results into Drive so every completed run is durable.
!rm -rf results/experiment6_dense_graph
!ln -s {PERSIST} results/experiment6_dense_graph
print("results persist to", PERSIST)
```

Cell 4 (code) — sanity-check the tests before spending GPU time:
```python
!pip install -q pytest
!python -m pytest tests/test_graph_edges.py tests/test_graph_models.py tests/test_graph_transformer_multi.py -q
```

Cell 5 (code) — Pass 1 (14 runs):
```python
!python scripts/run_dense_graph_grid.py --pass 1
```

Cell 6 (code) — Pass 2 (extends to 3 seeds, 42 cumulative):
```python
!python scripts/run_dense_graph_grid.py --pass 2
```

Cell 7 (code) — Pass 3 (adds GraphSAGE + R-GCN, 84 cumulative):
```python
!python scripts/run_dense_graph_grid.py --pass 3
```

Cell 8 (code) — figures:
```python
!python scripts/make_experiment6_figures.py
from IPython.display import Image, display
import glob
for p in sorted(glob.glob('results/experiment6_dense_graph/figures/*.png')):
    print(p); display(Image(p))
```

Cell 9 (code) — summary table:
```python
import pandas as pd
df = pd.read_csv('results/experiment6_dense_graph/results.csv')
summary = (df.groupby(['model','edges'])['mcc']
             .agg(['mean','std','count'])
             .sort_values('mean', ascending=False))
print(summary.to_string())
print("\nBaselines -- Exp1: +0.0922, Exp3 price-only: +0.0108, Exp4 sector GAT: +0.0278")
print("Constant predictors -- always-UP: acc 0.5129 / F1 0.6780, always-DOWN: acc 0.4871 / F1 0.0")
```

- [ ] **Step 2: Validate the notebook is well-formed**

Run:
```bash
python -c "
import json
nb = json.load(open('notebooks/MMGTFFF_dense_graph_experiment.ipynb'))
assert nb['nbformat'] == 4
print('cells:', len(nb['cells']))
assert len(nb['cells']) == 9
print('OK: notebook is valid nbformat v4 with 9 cells')
"
```
Expected: `OK: notebook is valid nbformat v4 with 9 cells`

- [ ] **Step 3: Commit**

```bash
git add notebooks/MMGTFFF_dense_graph_experiment.ipynb
git commit -m "Add Experiment 6 Colab notebook"
```

---

## Task 10: Local smoke validation (NOT the full grid)

**Files:**
- None created.

The full grid runs on Colab's GPU, not here. This task's only job is to
prove the wiring works end-to-end on real data before the user spends GPU
time on it — this project's established "validate locally before shipping"
practice, scoped down to minutes rather than hours.

- [ ] **Step 1: Run the whole test suite**

Run: `pytest tests/ -v`
Expected: every test passes, including the leakage tests from Tasks 1-3 and
the anti-collapse regressions from Task 4.

- [ ] **Step 2: Train exactly two cells end-to-end on real data**

This exercises the full path — edge cache, snapshot loading, edge
combination, model forward/backward, early stopping, result recording — on
the two models most likely to break (the adapter-wrapped GAT, and R-GCN,
the only model consuming `edge_type`).

```bash
python -c "
import sys, os, time; sys.path.insert(0,'.')
import pandas as pd
from scripts.run_dense_graph_grid import (train_one, record, MODELS, EXACT,
                                          FINAL, SECTOR_CSV, EDGE_CACHE, OUT_DIR)
from src.data.edge_cache import build_edge_cache, load_edge_cache
from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots

os.makedirs(OUT_DIR, exist_ok=True)
cache = load_edge_cache(EDGE_CACHE) if os.path.exists(EDGE_CACHE) else build_edge_cache(EXACT, FINAL, SECTOR_CSV, EDGE_CACHE)
print('edge cache days:', len(cache))

df = pd.read_parquet(EXACT)
tr = build_daily_snapshots(df,'train'); dv = build_daily_snapshots(df,'dev'); te = build_daily_snapshots(df,'test')
normalize_price_snapshots(tr, dv, te)
print('snapshots:', len(tr), len(dv), len(te))

for model_name in ['GAT','RGCN']:
    t0 = time.time()
    res = train_one(model_name, ['sector','correlation'], 0, (tr,dv,te), cache)
    print(f\"{model_name}: MCC={res['mcc']:+.4f} F1={res['f1']:.4f} AUC={res['auc']:.4f} n={res['n_samples']} ({time.time()-t0:.0f}s)\")
    assert res['n_samples'] == 3720, 'test split must stay comparable to Experiments 1/3/4/5'
print('OK: full pipeline runs end to end')
"
```

Expected: prints an MCC/F1/AUC line for each of GAT and RGCN, asserts
`n_samples == 3720`, and ends with `OK: full pipeline runs end to end`.

**If either cell exceeds MCC 0.15, stop and report it** — per the spec's
leakage tripwire, that is a suspected-leakage signal to be re-audited, not
a success to celebrate.

- [ ] **Step 3: Generate figures and confirm they render**

Run: `python scripts/make_experiment6_figures.py`
Expected: `graph_sector.png`, `graph_correlation.png`, `graph_comention.png`,
and `edge_density_over_time.png` exist under
`results/experiment6_dense_graph/figures/`. Confirm the three graph renders
show visibly different structure — sector as dense blobs, correlation and
co-mention as sparser — since that visual difference is the point of the
whole experiment.

- [ ] **Step 4: Commit the figures and edge cache metadata**

```bash
git add results/experiment6_dense_graph/figures/
git commit -m "Add Experiment 6 graph structure figures from smoke validation"
```

Do NOT commit `results.csv` or `edge_cache.pkl` from this smoke run — the
real grid runs on Colab, and a partial local results file would cause the
runner to skip those cells there.

---

## Task 11: Results write-up — DONE (2026-09-08)

> **Unblocked and completed.** The grid was run **locally on CPU**
> (`scripts/run_dense_graph_grid.py --pass 3`, all 84 cells, ~2 days
> wall-clock) rather than on Colab, so the block below is moot — there is
> no pending Colab run to wait for. Real `results.csv` and the write-up are
> committed; see `results/experiment6_dense_graph/EXPERIMENT6_DENSE_GRAPH.md`.

**Files:**
- Create: `results/experiment6_dense_graph/EXPERIMENT6_DENSE_GRAPH.md`

**This task cannot start until the user has run the notebook on Colab and
brought back `results.csv`.** The grid runs on Colab's GPU, not locally, so
there are no real numbers to write up before then. An implementer reaching
this task with no `results/experiment6_dense_graph/results.csv` present
should stop and report that the task is blocked, rather than writing a
document with placeholder or invented numbers.

- [ ] **Step 1: Write the results document**

Follow `results/experiment4_gat/EXPERIMENT4_GAT.md`'s structure. Fill every
number from the real `results.csv` — invent nothing. Required sections:

1. **Objective** — testing whether a richer graph beats the sector-only
   graph, the open question both Experiments 4 and 5 ended on.
2. **The MAN-SF non-reproduction caveat**, carried forward verbatim in
   substance: none of these graphs is MAN-SF's Wikidata-relation graph, and
   no result here may be described as reproducing or beating their 0.117.
3. **The three edge types**, with their measured properties: sector 381
   edges (9 complete cliques), correlation ~330 edges at 60d/top-5,
   co-mention live on 347 of 504 days.
4. **Results table** — mean ± std across seeds per (model, edge combo),
   against Experiment 1 (+0.0922), Experiment 3 price-only (+0.0108),
   Experiment 4 sector GAT (+0.0278), and both constant predictors
   (always-UP: accuracy 0.5129 / F1 0.6780; always-DOWN: accuracy 0.4871 /
   F1 0.0).
5. **Honest reading** — state plainly whether a richer graph helped. Any
   difference smaller than the seed spread must be called indistinguishable,
   not an improvement. If Experiment 4's +0.0278 does not survive its
   multi-seed re-run, say so explicitly; it is currently cited as this
   project's best graph result.
6. **Leakage audit** — state that the tripwire was checked and what it
   found.
7. **Disclosure** — uncontrolled differences (epoch budget, the 17
   uncovered join samples, correlation warm-up on the first ~30 training
   days).
8. **How we should move forward.**

- [ ] **Step 2: Commit**

```bash
git add results/experiment6_dense_graph/EXPERIMENT6_DENSE_GRAPH.md
git commit -m "Add Experiment 6 results write-up"
```

---

## Deliberately deferred: Pass 4 (text encoders)

The spec defines a fourth pass — `HierarchicalTweetEncoder` with FinBERT
and USE embeddings, concatenated into node features before the graph layers
— which **this plan does not implement**. This is a decomposition decision,
not an oversight:

- Pass 4 is defined in the spec as *"winning graph x text backends"*, and
  the winning graph is only known once Pass 3 completes. Building it now
  would mean guessing which of the 28 `(model, edge_combo)` cells to
  target, or running text across all of them — 5-10x the GPU cost for
  combinations most of which will be discarded.
- The text encoder is a substantial component in its own right (per-tweet
  embedding of ~106k tweets, disk caching, a hierarchical attention module
  ported from Experiment 3, and a fusion point inside each of four
  models). Folding it in would roughly double this plan's size while making
  the graph question — which is what the user asked for first — slower to
  answer.

**Follow-on plan required.** After Task 11 reports Pass 1-3 results, write
a second spec+plan covering Pass 4, targeting the winning cell as defined
in the spec's operational tie-break (highest mean test MCC across seeds;
fewer edge types wins ties within one standard deviation).
