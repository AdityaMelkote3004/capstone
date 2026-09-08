# Experiment 7 — Macro-Conditioned Micro Signal over a Tweet Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a graph learned from tweet semantics whose per-ticker text signal is FiLM-conditioned by a daily market-level event vector, and measure it with an 8-row ablation ladder that can tell a real graph contribution apart from market-direction prediction.

**Architecture:** A one-pass stream-filter pulls FNSPID news into two channels (macro = symbol-less market-wide headlines; micro = per-ticker headlines). FinBERT embeds both plus the existing tweet corpus into a disk cache, so the 40-run ladder pays that cost once. A daily macro vector generates FiLM parameters that transform each ticker's micro vector before it enters a GAT over a tweet-similarity graph. Every row of the ladder reports pooled *and* within-day metrics.

**Tech Stack:** PyTorch 2.11, torch_geometric 2.8 (`GATConv`), transformers 5.5 (`ProsusAI/finbert`), pandas, pyarrow, scikit-learn, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-event-conditioned-tweet-graph-design.md`

## Global Constraints

- Never describe results as reproducing or beating MAN-SF's GAT+price MCC of 0.117 or their headline 0.195 — measured on a Wikidata company-relation graph, not built here.
- All runs use `dataset/stocknet_protocol_a_exact.parquet`. Test `n_samples` must remain **3720**, matching Experiments 1/3/4/5/6.
- **5 seeds minimum** per ladder row. Report mean ± std always. A difference smaller than seed spread is "indistinguishable", never "an improvement".
- **Every results row reports pooled AND within-day metrics side by side.**
- **Leakage is a tested property.** Every builder takes an explicit cutoff and has a test asserting that corrupting data at or after the target date cannot change that date's output.
- **MCC > 0.15 tripwire stays armed** — any cell exceeding it is suspected leakage to re-audit, not a success.
- **English-only text.** All headlines are language-filtered before FinBERT; the reject rate is reported.
- Read-only files: `src/models/graph_model.py`, `src/models/graph_transformer.py`, `src/models/graph_models.py`, `src/data/graph_edges.py`, `src/models/price_encoder.py`, and all `results/experiment[1-6]*`.
- Results go in `results/experiment7_macro_conditioned/`. `metrics.json` schema matches Experiments 1/3/4/5/6.
- Follow repo style: module docstring, type hints on public functions.

---

## Task 0: Fix the `Price3` normalization poisoning

**Files:**
- Create: `src/data/price_fix.py`
- Test: `tests/test_price_fix.py`
- Modify: `src/data/graph_dataset.py` (add an opt-in parameter; default preserves current behaviour)

**Interfaces:**
- Produces: `winsorize_price_snapshots(snapshot_lists: list, limit: float = 10.0) -> dict` returning counts of clipped values per feature index. Consumed by Task 8's runner.
- Modifies: `normalize_price_snapshots(train_snapshots, *others, winsorize_limit: float | None = None)`.

**Root cause, already diagnosed — do not re-investigate:**

The three price features come from StockNet's own precomputed files (`price/preprocessed/{TICKER}.txt`, columns `data[3:6]`), ported in `preprocessing_notebooks/MMGTFFF_stocknet_EXACT_reproduction.ipynb`. Measured on the protocol-A parquet:

| Population | `Price3` std | `Price3` range |
|---|---|---|
| All 87 tickers | **154.002** | [−6495.0, +6500.0] |
| Excluding BRK-A and PCLN | **1.409** | [−44.32, +93.08] |

Only **255 rows of 26,619** exceed \|100\|: **BRK-A (252)** and **PCLN (3)** — the two highest-priced stocks in the universe. BRK-A's `Price1` is fine ([−0.013, +0.039]), so the defect is specific to the third ratio and scales with share price.

`normalize_price_snapshots` computes **one global mean/std across all tickers**, so these two tickers inflate std by 109× and compress the other 85 tickers' post-z-score range from ±3.53 to **±0.103** — a 34× loss of dynamic range on a third of the node features.

The fix is winsorization before normalization, not deletion: BRK-A is a legitimate universe member and dropping it would change `n_samples` away from 3720.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_price_fix.py`:

```python
import torch

from src.data.graph_dataset import DaySnapshot, normalize_price_snapshots
from src.data.price_fix import winsorize_price_snapshots


def _snapshot(values: list) -> DaySnapshot:
    """One snapshot with N tickers, 4 days, 3 price features."""
    seq = torch.tensor(values, dtype=torch.float32)
    n = seq.shape[0]
    return DaySnapshot(
        tickers=[f"T{i}" for i in range(n)],
        price_seq=seq,
        price_mask=torch.ones(n, 4),
        target=torch.zeros(n, dtype=torch.long),
        date="2014-01-02",
    )


def _normal_ticker() -> list:
    return [[0.1, 0.1, 0.5]] * 4


def _outlier_ticker() -> list:
    return [[0.1, 0.1, 6500.0]] * 4


def test_winsorize_clips_only_beyond_the_limit():
    snap = _snapshot([_normal_ticker(), _outlier_ticker()])
    winsorize_price_snapshots([[snap]], limit=10.0)
    assert snap.price_seq[0, 0, 2].item() == 0.5, "in-range values must be untouched"
    assert snap.price_seq[1, 0, 2].item() == 10.0, "out-of-range values clip to the limit"


def test_winsorize_clips_negative_side_too():
    snap = _snapshot([[[0.1, 0.1, -6495.0]] * 4])
    winsorize_price_snapshots([[snap]], limit=10.0)
    assert snap.price_seq[0, 0, 2].item() == -10.0


def test_winsorize_reports_what_it_clipped():
    snap = _snapshot([_normal_ticker(), _outlier_ticker()])
    report = winsorize_price_snapshots([[snap]], limit=10.0)
    assert report["clipped_per_feature"][2] == 4, "4 days of one outlier ticker"
    assert report["clipped_per_feature"][0] == 0
    assert report["total_values"] == 2 * 4 * 3


def test_winsorize_restores_dynamic_range_after_normalization():
    """THE regression: without winsorizing, one extreme ticker crushes every
    other ticker's third feature toward zero after z-scoring."""
    tickers = [_normal_ticker() for _ in range(20)]
    # give the normal tickers real spread on feature 2
    for i, t in enumerate(tickers):
        for d in range(4):
            t[d] = [0.1, 0.1, float(i) / 10.0]
    poisoned = _snapshot([*[list(t) for t in tickers], _outlier_ticker()])
    clean = _snapshot([*[list(t) for t in tickers], _outlier_ticker()])

    normalize_price_snapshots([poisoned])

    winsorize_price_snapshots([[clean]], limit=10.0)
    normalize_price_snapshots([clean])

    poisoned_spread = poisoned.price_seq[:20, :, 2].std().item()
    clean_spread = clean.price_seq[:20, :, 2].std().item()
    assert clean_spread > 10 * poisoned_spread, (
        f"winsorizing must restore spread: {clean_spread} vs {poisoned_spread}"
    )


def test_normalize_accepts_winsorize_limit_and_default_is_unchanged():
    a = _snapshot([_normal_ticker(), _outlier_ticker()])
    b = _snapshot([_normal_ticker(), _outlier_ticker()])
    normalize_price_snapshots([a])                          # default: no winsorizing
    normalize_price_snapshots([b], winsorize_limit=10.0)
    assert not torch.allclose(a.price_seq, b.price_seq)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_price_fix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.price_fix'`

- [ ] **Step 3: Write the implementation**

Create `src/data/price_fix.py`:

```python
"""Winsorization for the protocol-A price features.

`Price3` (StockNet's third precomputed price ratio, `data[5]` in
price/preprocessed/{TICKER}.txt) is price-scale-sensitive: across the 87-ticker
universe it ranges [-6495, +6500] with std 154.0, but excluding BRK-A and PCLN
-- the two highest-priced stocks, 255 rows of 26,619 -- it ranges [-44.3, +93.1]
with std 1.41.

Because `normalize_price_snapshots` computes one global mean/std across all
tickers, those two tickers inflate std 109x and compress every other ticker's
post-z-score range from +/-3.53 to +/-0.103. A third of the node features is
effectively constant zero as a result.

Winsorizing rather than dropping the tickers: BRK-A is a legitimate universe
member, and dropping it would move test n_samples off 3720 and break
comparability with Experiments 1/3/4/5/6.
"""

import torch


def winsorize_price_snapshots(snapshot_lists: list, limit: float = 10.0) -> dict:
    """Clips every price feature into [-limit, +limit], in place, across all
    DaySnapshot lists given. Returns a report of how many values were clipped
    per feature index, so the write-up can state exactly what was altered.

    Applied BEFORE normalization, and using a fixed absolute limit rather than a
    data-derived percentile, so the operation does not depend on which split it
    sees and cannot leak test-set information into training.
    """
    clipped = [0, 0, 0]
    total = 0
    for snap_list in snapshot_lists:
        for snap in snap_list:
            seq = snap.price_seq
            total += seq.numel()
            for feature in range(seq.shape[-1]):
                channel = seq[..., feature]
                clipped[feature] += int((channel.abs() > limit).sum().item())
            snap.price_seq = seq.clamp(min=-limit, max=limit)
    return {"clipped_per_feature": clipped, "total_values": total, "limit": limit}
```

Then modify `src/data/graph_dataset.py`'s `normalize_price_snapshots` signature and body:

```python
def normalize_price_snapshots(train_snapshots: list, *other_snapshot_lists: list,
                              winsorize_limit: float = None) -> None:
    """... (keep the existing docstring, then add:)

    When `winsorize_limit` is given, every price feature is clipped into
    [-limit, +limit] before train statistics are computed -- see
    src/data/price_fix.py for why the third feature needs this. The default of
    None preserves Experiments 4/5/6 behaviour exactly.
    """
    if winsorize_limit is not None:
        from src.data.price_fix import winsorize_price_snapshots
        winsorize_price_snapshots([train_snapshots, *other_snapshot_lists],
                                  limit=winsorize_limit)

    flat = torch.cat([snap.price_seq.reshape(-1, 3) for snap in train_snapshots], dim=0)
    # ... rest of the existing body unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_price_fix.py -v`
Expected: 5 passed

- [ ] **Step 5: Verify Experiments 4/5/6 are unaffected by default**

Run: `pytest tests/ -v`
Expected: all previously passing tests still pass — the new parameter defaults to `None`, so existing call sites behave identically.

- [ ] **Step 6: Measure the baseline shift**

```bash
python -c "
import sys, os, time; sys.path.insert(0,'.')
import pandas as pd
from scripts.run_dense_graph_grid import train_one, EXACT, EDGE_CACHE
from src.data.edge_cache import load_edge_cache
from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots

cache = load_edge_cache(EDGE_CACHE)
df = pd.read_parquet(EXACT)
tr = build_daily_snapshots(df,'train'); dv = build_daily_snapshots(df,'dev'); te = build_daily_snapshots(df,'test')
normalize_price_snapshots(tr, dv, te, winsorize_limit=10.0)
res = train_one('GAT', ['sector'], 0, (tr,dv,te), cache)
print('GAT[sector] seed=0 WITH Price3 fix:', {k: round(v,4) for k,v in res.items() if isinstance(v,float)})
print('Experiment 6 published value was MCC +0.0263')
assert res['n_samples'] == 3720
"
```

Record the number. **If it differs materially from +0.0263, Experiment 6's conclusions are affected and that is reported in Task 10's write-up** — the fix does not get quietly absorbed.

- [ ] **Step 7: Commit**

```bash
git add src/data/price_fix.py tests/test_price_fix.py src/data/graph_dataset.py
git commit -m "Fix Price3 normalization poisoning from two mega-priced tickers (Experiment 7 Task 0)"
```

---

## Task 1: FNSPID stream-filter with coverage report

**Files:**
- Create: `src/data/fnspid.py`
- Create: `scripts/fetch_fnspid.py`
- Test: `tests/test_fnspid.py`

**Interfaces:**
- Produces: `is_english(text: str, min_ascii_ratio: float = 0.9) -> bool`; `classify_row(symbol: str, universe: set) -> str` returning `"macro"`, `"micro"`, or `"skip"`; `stream_filter(line_iter, universe: set, start: str, end: str) -> tuple[list, list]` returning `(macro_rows, micro_rows)`; `coverage_report(rows: list, trading_days: list) -> dict`. Consumed by Tasks 3, 4.
- Output files: `data/processed/fnspid_macro.parquet`, `data/processed/fnspid_micro.parquet` with columns `["timestamp", "headline", "symbol"]`.

**Why a language heuristic instead of `langdetect`:** `langdetect` is not installed and adds a dependency for one boolean. FNSPID's non-English content observed in probing was Cyrillic, which an ASCII-ratio test rejects reliably. The reject rate is reported so the choice is auditable.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fnspid.py`:

```python
import pandas as pd

from src.data.fnspid import (classify_row, coverage_report, is_english,
                             stream_filter)

UNIVERSE = {"AAPL", "MSFT", "BRK-A"}

HEADER = ("Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,"
          "Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary\n")


def _row(ts: str, title: str, symbol: str = "") -> str:
    return f"{ts},{title},{symbol},http://u,Pub,Auth,,,,,\n"


def test_is_english_accepts_ascii_headlines():
    assert is_english("Market Wrap For April 23: S&P 500 Ends Winning Streak")


def test_is_english_rejects_cyrillic():
    assert not is_english("Бюджет-2005 принят Госдумой в третьем чтении")


def test_is_english_handles_empty_and_none():
    assert not is_english("")
    assert not is_english(None)


def test_classify_row_splits_macro_from_micro():
    assert classify_row("", UNIVERSE) == "macro"
    assert classify_row("AAPL", UNIVERSE) == "micro"
    assert classify_row("TSLA", UNIVERSE) == "skip", "outside the universe"


def test_stream_filter_keeps_only_the_window():
    lines = [HEADER,
             _row("2013-12-31 10:00:00 UTC", "too early"),
             _row("2014-06-02 10:00:00 UTC", "in window"),
             _row("2016-01-04 10:00:00 UTC", "too late")]
    macro, micro = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["in window"]
    assert micro == []


def test_stream_filter_routes_to_both_channels():
    lines = [HEADER,
             _row("2014-06-02 10:00:00 UTC", "market wide"),
             _row("2014-06-02 11:00:00 UTC", "apple news", "AAPL"),
             _row("2014-06-02 12:00:00 UTC", "other co", "TSLA")]
    macro, micro = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["market wide"]
    assert [r["headline"] for r in micro] == ["apple news"]
    assert micro[0]["symbol"] == "AAPL"


def test_stream_filter_drops_non_english():
    lines = [HEADER,
             _row("2014-06-02 10:00:00 UTC", "Бюджет принят Госдумой"),
             _row("2014-06-02 11:00:00 UTC", "normal headline")]
    macro, _ = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["normal headline"]


def test_stream_filter_survives_malformed_lines():
    """A 5.34 GB file will contain broken rows; one must not abort the pass."""
    lines = [HEADER,
             "garbage with no comma structure\n",
             ",,,,,,,,,,\n",
             _row("2014-06-02 10:00:00 UTC", "good row")]
    macro, _ = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert [r["headline"] for r in macro] == ["good row"]


def test_stream_filter_preserves_time_of_day():
    """Leakage control depends on the timestamp, not just the date."""
    lines = [HEADER, _row("2014-06-02 13:45:07 UTC", "timed")]
    macro, _ = stream_filter(iter(lines), UNIVERSE, "2014-01-01", "2015-12-31")
    assert macro[0]["timestamp"] == pd.Timestamp("2014-06-02 13:45:07")


def test_coverage_report_counts_days_and_flags_the_gate():
    days = [f"2014-01-{d:02d}" for d in range(1, 11)]
    rows = [{"timestamp": pd.Timestamp(f"{d} 10:00:00"), "headline": "h", "symbol": ""}
            for d in days[:6]]
    report = coverage_report(rows, days)
    assert report["days_covered"] == 6
    assert report["days_total"] == 10
    assert report["coverage_fraction"] == 0.6
    assert report["passes_gate"] is False, "0.6 is below the 0.70 gate"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fnspid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.fnspid'`

- [ ] **Step 3: Write the implementation**

Create `src/data/fnspid.py`:

```python
"""Streaming filter for the FNSPID financial-news corpus (Experiment 7).

FNSPID's Stock_news/All_external.csv is 5.34 GB / 15.7M rows and is sorted by
neither date nor symbol, so extracting the 2014-2015 window requires reading
the whole file once. This module does that in a single pass without ever
holding the raw file on disk, splitting rows into two channels:

  macro -- rows with an empty Stock_symbol (market-wide news, ~55% of rows)
  micro -- rows whose Stock_symbol is in the 87-ticker universe

Schema, read from the live header:
    Date, Article_title, Stock_symbol, Url, Publisher, Author,
    Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary

Only Date, Article_title and Stock_symbol are retained; Article and the four
summary columns are usually empty in the source.
"""

import csv

import pandas as pd

COVERAGE_GATE = 0.70


def is_english(text: str, min_ascii_ratio: float = 0.9) -> bool:
    """Cheap language gate: FinBERT is English-only and FNSPID contains
    non-English content (Cyrillic observed in probing). An ASCII-ratio test
    rejects that reliably without adding a dependency for one boolean."""
    if not text:
        return False
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / len(text) >= min_ascii_ratio


def classify_row(symbol: str, universe: set) -> str:
    """Routes a row to the macro channel (no symbol), the micro channel (a
    universe ticker), or discards it (a ticker outside the universe)."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return "macro"
    return "micro" if symbol in universe else "skip"


def stream_filter(line_iter, universe: set, start: str, end: str) -> tuple:
    """Single pass over FNSPID CSV lines, returning (macro_rows, micro_rows).

    Each row is {"timestamp": Timestamp, "headline": str, "symbol": str}.
    Malformed lines are skipped rather than raising -- a 5.34 GB corpus will
    contain broken rows and one must not abort a 34-minute pass.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    macro, micro = [], []

    reader = csv.reader(line_iter)
    try:
        next(reader)                      # header
    except StopIteration:
        return macro, micro

    for parts in reader:
        if len(parts) < 3:
            continue
        raw_date, title, symbol = parts[0], parts[1], parts[2]
        try:
            timestamp = pd.Timestamp(raw_date.replace(" UTC", ""))
        except (ValueError, TypeError):
            continue
        if pd.isna(timestamp) or not (start_ts <= timestamp < end_ts):
            continue
        if not is_english(title):
            continue

        channel = classify_row(symbol, universe)
        if channel == "skip":
            continue
        row = {"timestamp": timestamp, "headline": title,
               "symbol": (symbol or "").strip().upper()}
        (macro if channel == "macro" else micro).append(row)

    return macro, micro


def coverage_report(rows: list, trading_days: list) -> dict:
    """Per-trading-day headline counts, plus whether the channel clears the
    spec's 70% day-coverage gate. Below the gate the macro channel is
    redesigned rather than silently used."""
    counts: dict = {day: 0 for day in trading_days}
    for row in rows:
        day = row["timestamp"].strftime("%Y-%m-%d")
        if day in counts:
            counts[day] += 1
    covered = sum(1 for v in counts.values() if v > 0)
    total = max(len(trading_days), 1)
    fraction = covered / total
    return {
        "days_covered": covered,
        "days_total": len(trading_days),
        "coverage_fraction": fraction,
        "passes_gate": fraction >= COVERAGE_GATE,
        "headlines_total": len(rows),
        "per_day_counts": counts,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fnspid.py -v`
Expected: 10 passed

- [ ] **Step 5: Write the fetch script**

Create `scripts/fetch_fnspid.py`:

```python
"""Streams FNSPID once, filters to Experiment 7's window, writes two parquets.

Never stores the raw 5.34 GB file: the HTTP response is consumed line by line
and only matching rows are retained. Measured throughput ~2.7 MB/s, so expect
roughly 35 minutes. Safe to run in the background while other tasks proceed.
"""

import argparse
import io
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.fnspid import coverage_report, stream_filter

URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
       "Stock_news/All_external.csv")
EXACT = "dataset/stocknet_protocol_a_exact.parquet"
OUT_DIR = "data/processed"
START, END = "2014-01-01", "2015-12-31"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--local", default=None,
                    help="path to an already-downloaded CSV, for re-runs")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    exact = pd.read_parquet(EXACT, columns=["Ticker", "Target_Date"])
    universe = set(exact["Ticker"].unique())
    trading_days = sorted(pd.to_datetime(exact["Target_Date"]).dt.strftime("%Y-%m-%d").unique())
    print(f"universe: {len(universe)} tickers | trading days: {len(trading_days)}")

    if args.local:
        handle = open(args.local, "r", encoding="utf-8", errors="replace")
    else:
        print("streaming FNSPID (~5.34 GB, ~35 min at 2.7 MB/s)...", flush=True)
        response = urllib.request.urlopen(args.url, timeout=120)
        handle = io.TextIOWrapper(response, encoding="utf-8", errors="replace")

    with handle:
        macro, micro = stream_filter(handle, universe, START, END)

    print(f"macro rows: {len(macro):,} | micro rows: {len(micro):,}")
    for name, rows in (("macro", macro), ("micro", micro)):
        pd.DataFrame(rows).to_parquet(os.path.join(OUT_DIR, f"fnspid_{name}.parquet"))
        report = coverage_report(rows, trading_days)
        print(f"\n{name.upper()} CHANNEL")
        print(f"  headlines      : {report['headlines_total']:,}")
        print(f"  days covered   : {report['days_covered']}/{report['days_total']} "
              f"({100 * report['coverage_fraction']:.1f}%)")
        print(f"  passes 70% gate: {report['passes_gate']}")
        counts = [v for v in report["per_day_counts"].values() if v > 0]
        if counts:
            series = pd.Series(counts)
            print(f"  per-day counts : median={series.median():.0f} "
                  f"mean={series.mean():.0f} max={series.max():.0f}")

    print("\nIf the macro channel fails the 70% gate, STOP and redesign the "
          "macro channel per the spec rather than proceeding.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Launch the fetch in the background**

```bash
mkdir -p data/processed results/experiment7_macro_conditioned
nohup python -u scripts/fetch_fnspid.py > results/experiment7_macro_conditioned/fnspid_fetch.log 2>&1 &
echo "started; check results/experiment7_macro_conditioned/fnspid_fetch.log"
```

Continue to Task 2 while this runs — it is I/O-bound and needs ~35 minutes. Do not block on it.

- [ ] **Step 7: Commit**

```bash
git add src/data/fnspid.py scripts/fetch_fnspid.py tests/test_fnspid.py
git commit -m "Add FNSPID streaming filter with macro/micro split and coverage gate (Experiment 7)"
```

---

## Task 2: FinBERT embedding cache

**Files:**
- Create: `src/models/finbert_cache.py`
- Test: `tests/test_finbert_cache.py`

**Interfaces:**
- Produces: `cap_per_day(rows: list, n: int, cutoff_key: str = "timestamp") -> list` keeping the `n` most recent rows per calendar day; `embed_texts(texts: list, batch_size: int = 32) -> torch.Tensor` returning `(len(texts), 768)`; `EmbeddingCache` with `.get(key) -> torch.Tensor | None`, `.put(key, tensor)`, `.save(path)`, and classmethod `.load(path) -> EmbeddingCache`. Consumed by Tasks 3, 4.

**Why the cap is mandatory:** the spec projects ~3.9M macro headlines (~7,600 per trading day). FinBERT is BERT-base; at the 30–50 segments/sec obtainable from 16 CPU cores, embedding that corpus is tens of hours and would outrun the whole experiment. `N = 64` per day per channel is the starting default, set for real against Task 1's coverage report.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_finbert_cache.py`:

```python
import pandas as pd
import torch

from src.models.finbert_cache import EmbeddingCache, cap_per_day


def _rows(day: str, n: int) -> list:
    return [{"timestamp": pd.Timestamp(f"{day} {h:02d}:00:00"), "headline": f"h{h}"}
            for h in range(n)]


def test_cap_keeps_the_most_recent_n_per_day():
    rows = _rows("2014-06-02", 10)
    capped = cap_per_day(rows, n=3)
    assert len(capped) == 3
    assert {r["headline"] for r in capped} == {"h7", "h8", "h9"}


def test_cap_is_per_day_not_global():
    rows = _rows("2014-06-02", 5) + _rows("2014-06-03", 5)
    capped = cap_per_day(rows, n=2)
    assert len(capped) == 4, "2 per day across 2 days"


def test_cap_leaves_short_days_alone():
    rows = _rows("2014-06-02", 2)
    assert len(cap_per_day(rows, n=10)) == 2


def test_cap_on_empty_input():
    assert cap_per_day([], n=5) == []


def test_cache_roundtrips(tmp_path):
    cache = EmbeddingCache()
    cache.put("a", torch.ones(768))
    cache.put("b", torch.zeros(768))
    path = tmp_path / "emb.pt"
    cache.save(str(path))

    reloaded = EmbeddingCache.load(str(path))
    assert torch.allclose(reloaded.get("a"), torch.ones(768))
    assert reloaded.get("missing") is None


def test_cache_load_of_absent_file_is_empty_not_an_error(tmp_path):
    cache = EmbeddingCache.load(str(tmp_path / "nope.pt"))
    assert cache.get("anything") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_finbert_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.finbert_cache'`

- [ ] **Step 3: Write the implementation**

Create `src/models/finbert_cache.py`:

```python
"""FinBERT embedding with a disk cache, shared by Experiment 7's macro and
micro channels.

The 40-run ablation ladder must not re-embed text 40 times, and FinBERT on CPU
is the experiment's single largest fixed cost, so every embedding is computed
once and cached by a stable key.

`cap_per_day` is not an optimisation but a feasibility requirement: the macro
channel projects ~7,600 headlines per trading day, which at BERT-base CPU
throughput would take tens of hours to embed.
"""

import collections
import os

import torch

MODEL_NAME = "ProsusAI/finbert"
_model = None
_tokenizer = None


def _load_model():
    """Lazily loads FinBERT so importing this module stays cheap in tests."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModel, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def cap_per_day(rows: list, n: int, cutoff_key: str = "timestamp") -> list:
    """Keeps at most `n` rows per calendar day, the most recent ones by
    timestamp. Selection uses only timestamps, never labels."""
    by_day = collections.defaultdict(list)
    for row in rows:
        by_day[row[cutoff_key].strftime("%Y-%m-%d")].append(row)

    kept = []
    for day_rows in by_day.values():
        day_rows.sort(key=lambda r: r[cutoff_key])
        kept.extend(day_rows[-n:])
    return kept


def embed_texts(texts: list, batch_size: int = 32) -> torch.Tensor:
    """Mean-pools FinBERT's last hidden state over non-padding tokens,
    returning (len(texts), 768). Mean-pooling rather than [CLS] because
    FinBERT's [CLS] is tuned for its sentiment head, not general similarity --
    and Task 4 builds a cosine-similarity graph on these vectors."""
    if not texts:
        return torch.zeros((0, 768))

    model, tokenizer = _load_model()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start:start + batch_size]]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(pooled)
    return torch.cat(out, dim=0)


class EmbeddingCache:
    """Keyed store of 768-d vectors, persisted with torch.save."""

    def __init__(self, store: dict = None):
        self._store = store or {}

    def get(self, key: str):
        return self._store.get(key)

    def put(self, key: str, tensor: torch.Tensor) -> None:
        self._store[key] = tensor

    def __len__(self) -> int:
        return len(self._store)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._store, path)

    @classmethod
    def load(cls, path: str) -> "EmbeddingCache":
        """Missing file yields an empty cache, so a first run and a resumed run
        take the same code path."""
        if not os.path.exists(path):
            return cls()
        return cls(torch.load(path, weights_only=True))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_finbert_cache.py -v`
Expected: 6 passed

- [ ] **Step 5: Smoke-test real FinBERT once**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch, time
from src.models.finbert_cache import embed_texts
t0=time.time()
v = embed_texts(['Market Wrap: S&P 500 ends winning streak',
                 'Apple beats earnings expectations'])
print('shape', tuple(v.shape), 'finite', bool(torch.isfinite(v).all()), '%.1fs'%(time.time()-t0))
assert v.shape == (2, 768)
print('OK')
"
```

Expected: `shape (2, 768) finite True`, then `OK`. First run downloads the model.

- [ ] **Step 6: Commit**

```bash
git add src/models/finbert_cache.py tests/test_finbert_cache.py
git commit -m "Add FinBERT embedding cache with mandatory per-day headline cap (Experiment 7)"
```

---

## Task 3: Macro calendar and daily macro vectors

**Files:**
- Create: `dataset/macro_calendar.csv`
- Create: `src/data/macro_encoder.py`
- Test: `tests/test_macro_encoder.py`

**Interfaces:**
- Consumes: `EmbeddingCache`, `embed_texts`, `cap_per_day` (Task 2); `data/processed/fnspid_macro.parquet` (Task 1).
- Produces: `load_macro_calendar(path: str) -> dict` mapping `"YYYY-MM-DD" -> (fomc, cpi, nfp)` int flags; `market_open_utc(day: str) -> pd.Timestamp`; `build_macro_vectors(headlines: list, calendar: dict, trading_days: list, cache, lookback_days: int = 3, cap: int = 64) -> dict` mapping `"YYYY-MM-DD" -> torch.Tensor` of shape `(771,)`. Consumed by Tasks 6, 8.

The vector is `768` FinBERT dims + `3` calendar flags = **771-d raw**; the learned 768→32 projection lives inside the model (Task 6), so the cache stays projection-agnostic and the ladder can vary the projection without re-embedding.

- [ ] **Step 1: Create the macro calendar**

Create `dataset/macro_calendar.csv` with header `date,fomc,cpi,nfp`. FOMC meeting dates are the second day of each scheduled 2014–2015 meeting; CPI and NFP are BLS release dates. Enter the real published dates — these are public record and must not be invented:

```csv
date,fomc,cpi,nfp
2014-01-29,1,0,0
2014-03-19,1,0,0
2014-04-30,1,0,0
2014-06-18,1,0,0
2014-07-30,1,0,0
2014-09-17,1,0,0
2014-10-29,1,0,0
2014-12-17,1,0,0
2015-01-28,1,0,0
2015-03-18,1,0,0
2015-04-29,1,0,0
2015-06-17,1,0,0
2015-07-29,1,0,0
2015-09-17,1,0,0
2015-10-28,1,0,0
2015-12-16,1,0,0
```

Then append the 2014–2015 CPI release dates with `cpi=1` and the monthly NFP (Employment Situation) release dates with `nfp=1`, one row per date. Where a date already appears, set the additional flag on the existing row rather than adding a duplicate. A verification step below asserts the file is well-formed and has no duplicate dates.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_macro_encoder.py`:

```python
import pandas as pd
import torch

from src.data.macro_encoder import (build_macro_vectors, load_macro_calendar,
                                    market_open_utc)
from src.models.finbert_cache import EmbeddingCache


class FakeCache(EmbeddingCache):
    """Deterministic stand-in so tests never invoke FinBERT."""

    def get(self, key):
        return torch.full((768,), float(len(key) % 7))


def _headlines(spec: list) -> list:
    return [{"timestamp": pd.Timestamp(ts), "headline": h} for ts, h in spec]


CALENDAR = {"2014-06-02": (1, 0, 0)}
DAYS = ["2014-06-02", "2014-06-03"]


def test_market_open_is_0930_eastern_in_utc():
    # 2014-06-02 is EDT (UTC-4), so 09:30 ET == 13:30 UTC
    assert market_open_utc("2014-06-02") == pd.Timestamp("2014-06-02 13:30:00")


def test_market_open_handles_winter_standard_time():
    # 2014-01-02 is EST (UTC-5), so 09:30 ET == 14:30 UTC
    assert market_open_utc("2014-01-02") == pd.Timestamp("2014-01-02 14:30:00")


def test_load_macro_calendar(tmp_path):
    path = tmp_path / "cal.csv"
    path.write_text("date,fomc,cpi,nfp\n2014-06-02,1,0,1\n")
    cal = load_macro_calendar(str(path))
    assert cal["2014-06-02"] == (1, 0, 1)


def test_vector_shape_and_calendar_flags_are_appended():
    heads = _headlines([("2014-06-02 10:00:00", "pre-open news")])
    vectors = build_macro_vectors(heads, CALENDAR, DAYS, FakeCache())
    assert vectors["2014-06-02"].shape == (771,)
    assert vectors["2014-06-02"][-3:].tolist() == [1.0, 0.0, 0.0]


def test_leakage_headlines_at_or_after_market_open_are_excluded():
    """THE leakage test: news published after the bell cannot inform that day."""
    clean = _headlines([("2014-06-02 10:00:00", "before open")])
    poisoned = clean + _headlines([
        ("2014-06-02 14:00:00", "AFTER OPEN - must be ignored"),
        ("2014-06-02 20:00:00", "ALSO AFTER - must be ignored"),
    ])
    a = build_macro_vectors(clean, CALENDAR, DAYS, FakeCache())
    b = build_macro_vectors(poisoned, CALENDAR, DAYS, FakeCache())
    assert torch.allclose(a["2014-06-02"], b["2014-06-02"])


def test_lookback_window_is_respected():
    old = _headlines([("2014-05-01 10:00:00", "far too old")])
    vectors = build_macro_vectors(old, CALENDAR, DAYS, FakeCache(), lookback_days=3)
    # no qualifying headline -> text half is the zero default
    assert torch.allclose(vectors["2014-06-02"][:768], torch.zeros(768))


def test_days_with_no_headlines_still_produce_a_vector():
    vectors = build_macro_vectors([], CALENDAR, DAYS, FakeCache())
    assert set(vectors) == set(DAYS)
    for day in DAYS:
        assert vectors[day].shape == (771,)
        assert torch.isfinite(vectors[day]).all()


def test_calendar_flags_default_to_zero_for_unlisted_days():
    vectors = build_macro_vectors([], CALENDAR, DAYS, FakeCache())
    assert vectors["2014-06-03"][-3:].tolist() == [0.0, 0.0, 0.0]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_macro_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.macro_encoder'`

- [ ] **Step 4: Write the implementation**

Create `src/data/macro_encoder.py`:

```python
"""Daily macro event vectors for Experiment 7.

One vector per trading day, shared by every ticker in that day's cross-section:
FinBERT over market-wide headlines published strictly before the opening bell,
concatenated with FOMC/CPI/NFP release flags.

The pre-open cutoff is the leakage guard, and it is why Task 1 preserves
FNSPID's time-of-day rather than truncating to dates.
"""

import pandas as pd
import torch

from src.models.finbert_cache import cap_per_day

MACRO_DIM = 768
CALENDAR_DIM = 3


def market_open_utc(day: str) -> pd.Timestamp:
    """09:30 America/New_York on `day`, expressed as a naive UTC timestamp so it
    compares directly against FNSPID's UTC column. Handles EST/EDT."""
    local = pd.Timestamp(f"{day} 09:30:00", tz="America/New_York")
    return local.tz_convert("UTC").tz_localize(None)


def load_macro_calendar(path: str) -> dict:
    """Maps 'YYYY-MM-DD' -> (fomc, cpi, nfp) integer flags."""
    df = pd.read_csv(path)
    return {
        str(row["date"])[:10]: (int(row["fomc"]), int(row["cpi"]), int(row["nfp"]))
        for _, row in df.iterrows()
    }


def build_macro_vectors(headlines: list, calendar: dict, trading_days: list,
                        cache, lookback_days: int = 3, cap: int = 64) -> dict:
    """Builds {day: (771,) tensor} for every day in `trading_days`.

    A headline counts toward day `t` only if its timestamp is strictly before
    `t`'s market open and within `lookback_days` calendar days of it. Days with
    no qualifying headline get a zero text half rather than being dropped --
    the model owns the learned default, not this cache.
    """
    vectors = {}
    for day in trading_days:
        cutoff = market_open_utc(day)
        window_start = cutoff - pd.Timedelta(days=lookback_days)
        eligible = [h for h in headlines if window_start <= h["timestamp"] < cutoff]
        eligible = cap_per_day(eligible, n=cap)

        if eligible:
            stacked = torch.stack([cache.get(h["headline"]) for h in eligible])
            text = stacked.mean(dim=0)
        else:
            text = torch.zeros(MACRO_DIM)

        flags = torch.tensor(calendar.get(day, (0, 0, 0)), dtype=torch.float32)
        vectors[day] = torch.cat([text, flags])
    return vectors
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_macro_encoder.py -v`
Expected: 8 passed

- [ ] **Step 6: Verify the calendar file is well-formed**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from src.data.macro_encoder import load_macro_calendar
df = pd.read_csv('dataset/macro_calendar.csv')
assert list(df.columns) == ['date','fomc','cpi','nfp'], df.columns
assert not df['date'].duplicated().any(), 'duplicate dates - merge the flags onto one row'
d = pd.to_datetime(df['date'])
assert d.min() >= pd.Timestamp('2014-01-01') and d.max() <= pd.Timestamp('2015-12-31')
cal = load_macro_calendar('dataset/macro_calendar.csv')
print('calendar rows:', len(df))
print('FOMC:', df.fomc.sum(), '| CPI:', df.cpi.sum(), '| NFP:', df.nfp.sum())
assert df.fomc.sum() == 16, 'expected 16 FOMC meetings across 2014-2015'
assert 20 <= df.cpi.sum() <= 24, 'expected ~24 monthly CPI releases'
assert 20 <= df.nfp.sum() <= 24, 'expected ~24 monthly NFP releases'
print('OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add dataset/macro_calendar.csv src/data/macro_encoder.py tests/test_macro_encoder.py
git commit -m "Add macro event calendar and pre-open-cutoff daily macro vectors (Experiment 7)"
```

---

## Task 4: Micro signal encoder

**Files:**
- Create: `src/data/micro_encoder.py`
- Test: `tests/test_micro_encoder.py`

**Interfaces:**
- Consumes: `EmbeddingCache`, `cap_per_day` (Task 2); `market_open_utc` (Task 3); `dataset/final/stocknet_final_modeling_set.parquet`; `data/processed/fnspid_micro.parquet`.
- Produces: `load_ticker_tweets(final_parquet: str) -> dict` mapping `(ticker, "YYYY-MM-DD") -> list[str]`; `build_micro_embeddings(tweets_by_key, news_rows, tickers, trading_days, cache, tweet_lookback=5, cap=64) -> dict` mapping `(ticker, day) -> (tweet_vec, news_vec)`, **both `(768,)`, returned separately**. Consumed by Tasks 5, 6, 8.

**Interface requirement from the spec:** the tweet vector must be exposed separately from the news vector. Task 5 builds the graph from the tweet half alone so that graph structure and news content stay separately attributable; a single pooled vector would silently break the ablation ladder's claims.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_micro_encoder.py`:

```python
import pandas as pd
import torch

from src.data.micro_encoder import build_micro_embeddings, load_ticker_tweets
from src.models.finbert_cache import EmbeddingCache


class FakeCache(EmbeddingCache):
    def get(self, key):
        return torch.full((768,), float(len(str(key)) % 5))


TICKERS = ["AAPL", "MSFT"]
DAYS = ["2014-06-03", "2014-06-04"]


def _news(spec: list) -> list:
    return [{"timestamp": pd.Timestamp(ts), "headline": h, "symbol": s}
            for ts, h, s in spec]


def test_returns_tweet_and_news_vectors_separately():
    tweets = {("AAPL", "2014-06-02"): ["apple tweet"]}
    news = _news([("2014-06-03 10:00:00", "apple headline", "AAPL")])
    out = build_micro_embeddings(tweets, news, TICKERS, DAYS, FakeCache())
    tweet_vec, news_vec = out[("AAPL", "2014-06-03")]
    assert tweet_vec.shape == (768,) and news_vec.shape == (768,)
    assert not torch.allclose(tweet_vec, news_vec), (
        "the two channels must stay distinct - Task 5 builds the graph from "
        "the tweet half alone"
    )


def test_every_ticker_day_pair_is_present():
    out = build_micro_embeddings({}, [], TICKERS, DAYS, FakeCache())
    assert set(out) == {(t, d) for t in TICKERS for d in DAYS}


def test_missing_text_yields_zero_vectors_not_a_crash():
    """The 17 protocol-A samples uncovered by the join must degrade, not fail."""
    out = build_micro_embeddings({}, [], TICKERS, DAYS, FakeCache())
    tweet_vec, news_vec = out[("AAPL", "2014-06-03")]
    assert torch.allclose(tweet_vec, torch.zeros(768))
    assert torch.allclose(news_vec, torch.zeros(768))


def test_leakage_tweets_on_the_target_day_are_excluded():
    same_day = {("AAPL", "2014-06-03"): ["SAME DAY - must be ignored"]}
    a = build_micro_embeddings({}, [], TICKERS, DAYS, FakeCache())
    b = build_micro_embeddings(same_day, [], TICKERS, DAYS, FakeCache())
    assert torch.allclose(a[("AAPL", "2014-06-03")][0],
                          b[("AAPL", "2014-06-03")][0])


def test_leakage_news_after_market_open_is_excluded():
    before = _news([("2014-06-03 10:00:00", "before open", "AAPL")])
    after = before + _news([("2014-06-03 18:00:00", "AFTER OPEN", "AAPL")])
    a = build_micro_embeddings({}, before, TICKERS, DAYS, FakeCache())
    b = build_micro_embeddings({}, after, TICKERS, DAYS, FakeCache())
    assert torch.allclose(a[("AAPL", "2014-06-03")][1],
                          b[("AAPL", "2014-06-03")][1])


def test_news_is_routed_to_the_right_ticker():
    news = _news([("2014-06-03 10:00:00", "apple only", "AAPL")])
    out = build_micro_embeddings({}, news, TICKERS, DAYS, FakeCache())
    assert not torch.allclose(out[("AAPL", "2014-06-03")][1], torch.zeros(768))
    assert torch.allclose(out[("MSFT", "2014-06-03")][1], torch.zeros(768))


def test_tweet_lookback_window_is_respected():
    old = {("AAPL", "2014-05-01"): ["far too old"]}
    out = build_micro_embeddings(old, [], TICKERS, DAYS, FakeCache(), tweet_lookback=5)
    assert torch.allclose(out[("AAPL", "2014-06-03")][0], torch.zeros(768))


def test_load_ticker_tweets_on_real_data():
    tweets = load_ticker_tweets("dataset/final/stocknet_final_modeling_set.parquet")
    assert len(tweets) > 10000
    key = next(iter(tweets))
    assert isinstance(key, tuple) and len(key) == 2
    assert isinstance(tweets[key], list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_micro_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.micro_encoder'`

- [ ] **Step 3: Write the implementation**

Create `src/data/micro_encoder.py`:

```python
"""Per-(ticker, day) micro signal for Experiment 7.

Two channels, deliberately kept separate:

  tweet_vec -- FinBERT over the ticker's tweets in a window ending at t-1
  news_vec  -- FinBERT over the ticker's FNSPID headlines before t's open

They are returned as a pair rather than pooled because Task 5 builds the
tweet-similarity graph from the tweet half ALONE. Pooling here would make graph
structure and news content inseparable, and the ablation ladder would silently
stop isolating what it claims to isolate.
"""

import collections

import pandas as pd
import torch

from src.data.macro_encoder import market_open_utc
from src.models.finbert_cache import cap_per_day

MICRO_DIM = 768


def load_ticker_tweets(final_parquet: str) -> dict:
    """Maps (ticker, 'YYYY-MM-DD') -> list of tweet segment strings, pooling
    Company_Texts and Event_Texts. Rows with no text are skipped."""
    df = pd.read_parquet(final_parquet,
                         columns=["Date", "Ticker", "Company_Texts", "Event_Texts"])
    df["Date"] = pd.to_datetime(df["Date"])
    out: dict = collections.defaultdict(list)
    for date, ticker, company, event in zip(df["Date"], df["Ticker"],
                                            df["Company_Texts"], df["Event_Texts"]):
        key = (ticker, date.strftime("%Y-%m-%d"))
        for blob in (company, event):
            if blob is None or len(str(blob)) <= 2:
                continue
            out[key].extend(s for s in str(blob).split("[SEP]") if len(s.strip()) > 2)
    return dict(out)


def _mean_or_zero(texts: list, cache) -> torch.Tensor:
    if not texts:
        return torch.zeros(MICRO_DIM)
    vectors = [cache.get(t) for t in texts]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return torch.zeros(MICRO_DIM)
    return torch.stack(vectors).mean(dim=0)


def build_micro_embeddings(tweets_by_key: dict, news_rows: list, tickers: list,
                           trading_days: list, cache, tweet_lookback: int = 5,
                           cap: int = 64) -> dict:
    """Builds {(ticker, day): (tweet_vec, news_vec)} for every pair.

    Tweets use a window ending at t-1 (strictly excluding the target day);
    news uses everything published before t's market open. Missing text yields
    zero vectors -- the 17 protocol-A samples uncovered by the tweet join must
    degrade to an empty neighbourhood, never crash and never be dropped.
    """
    news_by_symbol: dict = collections.defaultdict(list)
    for row in news_rows:
        news_by_symbol[row["symbol"]].append(row)

    out = {}
    for day in trading_days:
        day_ts = pd.Timestamp(day)
        cutoff = market_open_utc(day)
        window_start = day_ts - pd.Timedelta(days=tweet_lookback)

        for ticker in tickers:
            texts = []
            for offset in range(1, tweet_lookback + 1):
                past = (day_ts - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                if pd.Timestamp(past) >= window_start:
                    texts.extend(tweets_by_key.get((ticker, past), []))

            eligible_news = [n for n in news_by_symbol.get(ticker, [])
                             if n["timestamp"] < cutoff]
            eligible_news = cap_per_day(eligible_news, n=cap)

            out[(ticker, day)] = (
                _mean_or_zero(texts[-cap:], cache),
                _mean_or_zero([n["headline"] for n in eligible_news], cache),
            )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_micro_encoder.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/micro_encoder.py tests/test_micro_encoder.py
git commit -m "Add micro signal encoder exposing tweet and news channels separately (Experiment 7)"
```

---

## Task 5: Tweet-similarity graph builder

**Files:**
- Create: `src/data/tweet_graph.py`
- Test: `tests/test_tweet_graph.py`

**Interfaces:**
- Consumes: `build_micro_embeddings` output (Task 4); `combine_edges` from `src.data.graph_edges` (read-only, reused unchanged).
- Produces: `build_tweet_edges(micro_embeddings: dict, tickers: list, target_date: str, k: int = 5) -> set` returning alphabetically-sorted `(a, b)` tuples. Consumed by Task 8.

Top-k rather than a cosine threshold, for the reason Experiment 6 documented: a threshold produced 57% density and rebuilt the over-smoothing collapse diagnosed in Experiment 4. Top-k makes density a controlled parameter.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tweet_graph.py`:

```python
import torch

from src.data.tweet_graph import build_tweet_edges

DAY = "2014-06-03"
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _embeddings(vectors: dict) -> dict:
    """vectors: ticker -> 768-d tweet vector. News half is unused here."""
    return {(t, DAY): (v, torch.zeros(768)) for t, v in vectors.items()}


def _aligned(seed: float) -> torch.Tensor:
    v = torch.zeros(768)
    v[0] = seed
    v[1] = 1.0
    return v


def test_connects_the_similar_tickers():
    emb = _embeddings({
        "AAA": _aligned(1.0), "BBB": _aligned(1.0),      # identical direction
        "CCC": -_aligned(1.0), "DDD": -_aligned(1.0),    # opposite direction
    })
    edges = build_tweet_edges(emb, TICKERS, DAY, k=1)
    assert ("AAA", "BBB") in edges
    assert ("CCC", "DDD") in edges


def test_edges_are_sorted_tuples_without_duplicates():
    emb = _embeddings({t: torch.randn(768) for t in TICKERS})
    edges = build_tweet_edges(emb, TICKERS, DAY, k=2)
    for a, b in edges:
        assert a < b
    assert len(edges) == len(set(edges))


def test_no_self_edges():
    emb = _embeddings({t: torch.randn(768) for t in TICKERS})
    edges = build_tweet_edges(emb, TICKERS, DAY, k=3)
    assert all(a != b for a, b in edges)


def test_top_k_caps_degree():
    emb = _embeddings({t: torch.randn(768) for t in TICKERS})
    k = 2
    edges = build_tweet_edges(emb, TICKERS, DAY, k=k)
    assert len(edges) <= k * len(TICKERS)


def test_tickers_with_no_tweets_contribute_no_edges():
    emb = _embeddings({
        "AAA": _aligned(1.0), "BBB": _aligned(1.0),
        "CCC": torch.zeros(768), "DDD": torch.zeros(768),   # no tweets
    })
    edges = build_tweet_edges(emb, TICKERS, DAY, k=3)
    assert all("CCC" not in e and "DDD" not in e for e in edges)


def test_empty_day_returns_empty_set_not_an_error():
    edges = build_tweet_edges({}, TICKERS, DAY, k=5)
    assert edges == set()


def test_uses_only_the_tweet_half_not_the_news_half():
    """The graph must be attributable to tweets alone - if news leaked into
    the similarity, ablation rows 3 and 4 stop being separable."""
    same_tweets = _aligned(1.0)
    emb = {
        ("AAA", DAY): (same_tweets, torch.randn(768)),
        ("BBB", DAY): (same_tweets, torch.randn(768) * 100),   # wildly different news
        ("CCC", DAY): (-same_tweets, torch.randn(768)),
        ("DDD", DAY): (-same_tweets, torch.randn(768)),
    }
    edges = build_tweet_edges(emb, TICKERS, DAY, k=1)
    assert ("AAA", "BBB") in edges, "identical tweets must still connect"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tweet_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.tweet_graph'`

- [ ] **Step 3: Write the implementation**

Create `src/data/tweet_graph.py`:

```python
"""Tweet-similarity edges for Experiment 7.

The first graph in this project derived from information that is neither
price-derived nor static: two tickers connect when what people said about them
recently was semantically similar.

Built from the TWEET half of the micro embeddings only, never the news half, so
that graph structure and news content remain separately attributable in the
ablation ladder. Leakage safety comes from Task 4's windowing -- the tweet
vector for day t already excludes day t.
"""

import torch


def build_tweet_edges(micro_embeddings: dict, tickers: list, target_date: str,
                      k: int = 5) -> set:
    """Connects each ticker to its top-`k` most cosine-similar partners by
    tweet embedding, returning undirected alphabetically-sorted pairs.

    Top-k rather than a similarity threshold: Experiment 6 measured a threshold
    at 57% density, which rebuilt the over-smoothing collapse diagnosed in
    Experiment 4. Top-k makes density a controlled parameter.

    Tickers with no tweets that day carry a zero vector and are excluded --
    a zero vector has no meaningful direction to compare.
    """
    present, vectors = [], []
    for ticker in tickers:
        entry = micro_embeddings.get((ticker, target_date))
        if entry is None:
            continue
        tweet_vec = entry[0]
        if float(tweet_vec.abs().sum()) == 0.0:
            continue
        present.append(ticker)
        vectors.append(tweet_vec)

    if len(present) < 2:
        return set()

    matrix = torch.nn.functional.normalize(torch.stack(vectors), dim=1)
    similarity = matrix @ matrix.t()
    similarity.fill_diagonal_(float("-inf"))        # never self-edge

    edges = set()
    neighbours = min(k, len(present) - 1)
    top = similarity.topk(neighbours, dim=1).indices
    for i, ticker in enumerate(present):
        for j in top[i].tolist():
            edges.add(tuple(sorted((ticker, present[j]))))
    return edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tweet_graph.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/tweet_graph.py tests/test_tweet_graph.py
git commit -m "Add tweet-similarity graph builder using the tweet channel only (Experiment 7)"
```

---

## Task 6: `MacroConditionedGraphNet`

**Files:**
- Create: `src/models/macro_conditioned.py`
- Test: `tests/test_macro_conditioned.py`

**Interfaces:**
- Consumes: `PriceEncoder` from `src.models.price_encoder` (read-only).
- Produces: `MacroConditionedGraphNet(price_input_dim=3, hidden_dim=64, micro_dim=768, macro_dim=771, micro_proj=32, heads=4, dropout=0.2, use_micro=True, use_graph=True, conditioning="micro")` with `.forward(price_seq, price_mask, edge_index, micro, macro) -> (N, 2)`, and attribute `.last_film` holding `(gamma, beta)` from the most recent forward pass. Consumed by Tasks 8, 9.

The single class covers every ablation row via its flags, so the ladder varies configuration rather than code:

| Row | `use_micro` | `use_graph` | `conditioning` |
|---|---|---|---|
| 0 macro-only | False | False | `"micro"` (no per-ticker input; see below) |
| 1 price-only | False | False | `"none"` |
| 2 micro, no graph | True | False | `"none"` |
| 3 graph, no conditioning | True | True | `"none"` |
| 4 full model | True | True | `"micro"` |
| 5 shuffled macro | True | True | `"micro"` (runner permutes macro) |
| 6 attention site | True | True | `"attention"` |
| 7 sector graph | True | True | `"micro"` (runner supplies sector edges) |

Row 0 is expressed by a separate `macro_only=True` flag that drops price and micro entirely, so the model *cannot* discriminate between tickers on a day — that is the point of the control.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_macro_conditioned.py`:

```python
from itertools import combinations

import torch

from src.models.macro_conditioned import MacroConditionedGraphNet


def _clique(n: int) -> torch.Tensor:
    src, dst = [], []
    for a, b in combinations(range(n), 2):
        src.extend([a, b])
        dst.extend([b, a])
    return torch.tensor([src, dst], dtype=torch.long)


def _inputs(n: int = 5):
    return (torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n),
            torch.randn(n, 768), torch.randn(771))


def test_forward_shape():
    model = MacroConditionedGraphNet(hidden_dim=16)
    logits = model(*_inputs(5))
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_empty_edge_index_still_runs():
    model = MacroConditionedGraphNet(hidden_dim=16)
    price, mask, _, micro, macro = _inputs(3)
    logits = model(price, mask, torch.empty((2, 0), dtype=torch.long), micro, macro)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_macro_only_gives_every_ticker_the_same_prediction():
    """Row 0's whole purpose: it CANNOT discriminate between stocks."""
    model = MacroConditionedGraphNet(hidden_dim=16, macro_only=True)
    model.eval()
    price, mask, edges, micro, macro = _inputs(6)
    with torch.no_grad():
        logits = model(price, mask, edges, micro, macro)
    assert logits.shape == (6, 2)
    assert torch.allclose(logits, logits[0].expand_as(logits), atol=1e-6)


def test_conditioning_none_ignores_macro():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="none")
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, micro, torch.randn(771))
    assert torch.allclose(a, b), "conditioning='none' must not read macro"


def test_micro_conditioning_changes_the_output():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="micro")
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, micro, torch.randn(771) * 5)
    assert not torch.allclose(a, b)


def test_film_parameters_are_exposed_for_the_explainability_readout():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="micro")
    model(*_inputs(4))
    gamma, beta = model.last_film
    assert gamma.shape == (model.micro_proj,)
    assert beta.shape == (model.micro_proj,)


def test_use_micro_false_ignores_micro():
    model = MacroConditionedGraphNet(hidden_dim=16, use_micro=False)
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, torch.randn(4, 768), macro)
    assert torch.allclose(a, b)


def test_use_graph_false_ignores_edges():
    model = MacroConditionedGraphNet(hidden_dim=16, use_graph=False)
    model.eval()
    price, mask, _, micro, macro = _inputs(5)
    with torch.no_grad():
        a = model(price, mask, _clique(5), micro, macro)
        b = model(price, mask, torch.empty((2, 0), dtype=torch.long), micro, macro)
    assert torch.allclose(a, b)


def test_attention_conditioning_reads_macro():
    model = MacroConditionedGraphNet(hidden_dim=16, conditioning="attention")
    model.eval()
    price, mask, edges, micro, macro = _inputs(4)
    with torch.no_grad():
        a = model(price, mask, edges, micro, macro)
        b = model(price, mask, edges, micro, torch.randn(771) * 5)
    assert not torch.allclose(a, b)


def test_does_not_collapse_on_a_clique_after_training():
    """Experiment 4's over-smoothing regression, ported."""
    torch.manual_seed(0)
    n = 8
    model = MacroConditionedGraphNet(hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    price, mask, edges = torch.randn(n, 4, 3), torch.ones(n, 4), _clique(n)
    micro, macro = torch.randn(n, 768), torch.randn(771)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(
            model(price, mask, edges, micro, macro), target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(price, mask, edges, micro, macro)
    assert logits.std(dim=0).mean().item() > 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_macro_conditioned.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.macro_conditioned'`

- [ ] **Step 3: Write the implementation**

Create `src/models/macro_conditioned.py`:

```python
"""Experiment 7's model: a daily macro event vector FiLM-conditions each
ticker's micro (tweet + news) representation before a GAT mixes tickers over a
tweet-similarity graph.

The conditioning claim is semantic: an earnings headline means something
different during a tightening cycle than during easing. Whether the attention
logits are the better conditioning site is an empirical question, tested as
`conditioning="attention"` rather than assumed.

`gamma` and `beta` are shared across every ticker on a given day. That is
deliberate -- the explainability readout plots one gamma per trading day
against the macro calendar -- but it also means macro conditioning cannot by
itself reorder tickers within a day, which is why the ablation ladder carries a
macro-only control (`macro_only=True`) and reports within-day metrics.

Residual connections wrap each GAT layer for the over-smoothing reason
documented in results/experiment4_gat/EXPERIMENT4_GAT.md.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from src.models.price_encoder import PriceEncoder


class MacroConditionedGraphNet(nn.Module):
    """One class covering every row of Experiment 7's ablation ladder."""

    def __init__(self, price_input_dim: int = 3, hidden_dim: int = 64,
                 micro_dim: int = 768, macro_dim: int = 771, micro_proj: int = 32,
                 heads: int = 4, dropout: float = 0.2, use_micro: bool = True,
                 use_graph: bool = True, conditioning: str = "micro",
                 macro_only: bool = False):
        super().__init__()
        if conditioning not in ("none", "micro", "attention"):
            raise ValueError(f"unknown conditioning: {conditioning}")

        self.use_micro = use_micro
        self.use_graph = use_graph
        self.conditioning = conditioning
        self.macro_only = macro_only
        self.micro_proj = micro_proj
        self.last_film = None

        self.price_encoder = PriceEncoder(price_input_dim, hidden_dim)
        self.micro_projection = nn.Linear(micro_dim, micro_proj)
        self.micro_default = nn.Parameter(torch.zeros(micro_proj))
        self.film = nn.Sequential(
            nn.Linear(macro_dim, 64), nn.ReLU(), nn.Linear(64, 2 * micro_proj),
        )
        self.attention_film = nn.Sequential(
            nn.Linear(macro_dim, 32), nn.ReLU(), nn.Linear(32, 2),
        )

        node_dim = hidden_dim + (micro_proj if use_micro else 0)
        self.gat1 = GATConv(node_dim, node_dim, heads=heads, concat=False,
                            dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(node_dim, node_dim, heads=heads, concat=False,
                            dropout=dropout, add_self_loops=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(node_dim, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2),
        )
        self.macro_head = nn.Sequential(
            nn.Linear(macro_dim, 32), nn.ReLU(), nn.Linear(32, 2),
        )

    def forward(self, price_seq: torch.Tensor, price_mask: torch.Tensor,
                edge_index: torch.Tensor, micro: torch.Tensor,
                macro: torch.Tensor) -> torch.Tensor:
        n_nodes = price_seq.shape[0]

        if self.macro_only:
            # No per-ticker input at all: every ticker on this day receives an
            # identical prediction, so this row cannot discriminate stocks.
            self.last_film = None
            return self.macro_head(macro).unsqueeze(0).expand(n_nodes, 2)

        gamma = beta = None
        if self.conditioning in ("micro", "attention"):
            raw = self.film(macro)
            gamma, beta = raw[:self.micro_proj], raw[self.micro_proj:]
            self.last_film = (gamma.detach(), beta.detach())
        else:
            self.last_film = None

        h = self.price_encoder(price_seq, price_mask)

        if self.use_micro:
            projected = self.micro_projection(micro)
            empty = micro.abs().sum(dim=1, keepdim=True) == 0
            projected = torch.where(empty, self.micro_default.expand_as(projected),
                                     projected)
            if self.conditioning == "micro":
                projected = gamma.unsqueeze(0) * projected + beta.unsqueeze(0)
            h = torch.cat([h, projected], dim=1)

        if self.use_graph:
            scale = shift = None
            if self.conditioning == "attention":
                params = self.attention_film(macro)
                scale, shift = params[0], params[1]

            message = torch.relu(self.gat1(h, edge_index))
            if scale is not None:
                message = scale * message + shift
            h = h + message
            h = self.dropout(h)
            message = torch.relu(self.gat2(h, edge_index))
            if scale is not None:
                message = scale * message + shift
            h = h + message

        return self.classifier(h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_macro_conditioned.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/macro_conditioned.py tests/test_macro_conditioned.py
git commit -m "Add MacroConditionedGraphNet covering all 8 ablation rows (Experiment 7)"
```

---

## Task 7: Within-day metrics

**Files:**
- Create: `src/utils/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `pooled_metrics(targets, probs) -> dict` with keys `accuracy, f1, mcc, auc`; `within_day_metrics(targets, probs, days) -> dict` with keys `within_day_mcc, within_day_accuracy, days_scored, days_skipped`; `market_direction_oracle(targets, days) -> dict`. Consumed by Tasks 8, 9, 10.

**Why this task exists:** measured on the protocol-A splits, **48.3% of test label variance is between-day**, 32 of 64 test days are >80% one-directional, and an oracle that predicts only market direction — with zero ability to tell stocks apart — scores pooled MCC **+0.6248** / accuracy **0.8116**. Pooled MCC therefore conflates "predicts market direction" with "discriminates stocks". The graph exists to do the second.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import numpy as np

from src.utils.metrics import (market_direction_oracle, pooled_metrics,
                               within_day_metrics)


def test_pooled_metrics_on_a_perfect_predictor():
    targets = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    m = pooled_metrics(targets, probs)
    assert m["accuracy"] == 1.0
    assert m["mcc"] == 1.0
    assert m["auc"] == 1.0


def test_pooled_metrics_single_class_auc_falls_back_to_half():
    targets = np.array([1, 1, 1])
    probs = np.array([0.6, 0.7, 0.8])
    assert pooled_metrics(targets, probs)["auc"] == 0.5


def test_within_day_skips_single_class_days_and_counts_them():
    targets = np.array([1, 1, 0, 1])
    probs = np.array([0.9, 0.8, 0.2, 0.7])
    days = np.array(["d1", "d1", "d2", "d2"])
    m = within_day_metrics(targets, probs, days)
    assert m["days_scored"] == 1, "d1 is single-class and must be skipped"
    assert m["days_skipped"] == 1


def test_within_day_mcc_is_perfect_when_each_day_is_ranked_right():
    targets = np.array([0, 1, 0, 1])
    probs = np.array([0.2, 0.8, 0.3, 0.9])
    days = np.array(["d1", "d1", "d2", "d2"])
    assert within_day_metrics(targets, probs, days)["within_day_mcc"] == 1.0


def test_within_day_exposes_a_market_direction_only_predictor():
    """THE point of this module: a predictor that only knows the day's
    direction scores high pooled but zero within-day."""
    targets = np.array([1, 1, 1, 0, 0, 0])
    days = np.array(["d1", "d1", "d1", "d2", "d2", "d2"])
    probs = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])   # identical within each day
    assert pooled_metrics(targets, probs)["accuracy"] == 1.0
    m = within_day_metrics(targets, probs, days)
    assert m["days_scored"] == 0, "no day has both classes, so nothing is scored"


def test_within_day_zero_when_predictions_are_flat_within_mixed_days():
    targets = np.array([0, 1, 0, 1])
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    days = np.array(["d1", "d1", "d2", "d2"])
    assert within_day_metrics(targets, probs, days)["within_day_mcc"] == 0.0


def test_market_direction_oracle_matches_the_measured_test_value():
    targets = np.array([1, 1, 1, 0, 0, 1])
    days = np.array(["d1", "d1", "d1", "d2", "d2", "d2"])
    m = market_direction_oracle(targets, days)
    # d1 majority UP (3/3 correct); d2 majority DOWN (2/3 correct)
    assert m["accuracy"] == 5 / 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.metrics'`

- [ ] **Step 3: Write the implementation**

Create `src/utils/metrics.py`:

```python
"""Pooled and within-day metrics for Experiment 7.

Measured on the protocol-A splits, 48.3% of test-set label variance is
BETWEEN-day: on a given trading day the ~58 tickers in the cross-section mostly
move together, and 32 of 64 test days are more than 80% one-directional. An
oracle knowing only each day's market direction -- with no ability to tell
stocks apart -- scores pooled MCC +0.6248 and accuracy 0.8116, far above every
published number on this benchmark.

Pooled MCC therefore rewards two different abilities at once. The graph exists
to discriminate between stocks, so every result must be reported with within-day
metrics beside the pooled ones.

This is not leakage: market direction is legitimately part of the task as
StockNet defines it, and all published numbers are pooled. It is a
measurement-validity issue.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)


def pooled_metrics(targets, probs) -> dict:
    """Standard metrics over all samples, matching Experiments 1/3/4/5/6."""
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, preds)),
        "auc": float(roc_auc_score(targets, probs)) if len(set(targets.tolist())) > 1 else 0.5,
    }


def within_day_metrics(targets, probs, days) -> dict:
    """MCC and accuracy computed inside each day's cross-section, then averaged.

    Days with a single class present are skipped (MCC is undefined there) and
    counted, so the reader can see how much of the test set the within-day
    number actually covers.

    Predictions are compared against each day's own median probability rather
    than a fixed 0.5, so a model that shifts every ticker on a day in the same
    direction gains nothing here -- which is exactly what separates stock
    discrimination from market-direction prediction.
    """
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    days = np.asarray(days)

    mccs, accuracies, skipped = [], [], 0
    for day in np.unique(days):
        mask = days == day
        day_targets = targets[mask]
        if len(set(day_targets.tolist())) < 2:
            skipped += 1
            continue
        day_probs = probs[mask]
        day_preds = (day_probs >= np.median(day_probs)).astype(int)
        mccs.append(matthews_corrcoef(day_targets, day_preds))
        accuracies.append(accuracy_score(day_targets, day_preds))

    return {
        "within_day_mcc": float(np.mean(mccs)) if mccs else 0.0,
        "within_day_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "days_scored": len(mccs),
        "days_skipped": skipped,
    }


def market_direction_oracle(targets, days) -> dict:
    """Upper bound for a predictor that knows each day's majority direction and
    nothing else. Reported as a reference row so readers can see what pooled
    MCC is capable of rewarding without any stock discrimination at all."""
    targets = np.asarray(targets)
    days = np.asarray(days)
    preds = np.zeros_like(targets)
    for day in np.unique(days):
        mask = days == day
        preds[mask] = int(targets[mask].mean() >= 0.5)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "mcc": float(matthews_corrcoef(targets, preds)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 7 passed

- [ ] **Step 5: Reproduce the spec's oracle figure on real data**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from src.utils.metrics import market_direction_oracle
df = pd.read_parquet('dataset/stocknet_protocol_a_exact.parquet')
te = df[df.Split=='test']
m = market_direction_oracle(te.Target.values, te.Target_Date.astype(str).values)
print('market-direction oracle on test:', {k: round(v,4) for k,v in m.items()})
assert abs(m['mcc'] - 0.6248) < 0.01, 'must match the spec figure'
print('OK - matches the spec')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/utils/metrics.py tests/test_metrics.py
git commit -m "Add within-day metrics and market-direction oracle (Experiment 7)"
```

---

## Task 8: Ablation ladder runner

**Files:**
- Create: `scripts/run_experiment7_ladder.py`
- Create: `scripts/cache_experiment7_embeddings.py`

**Interfaces:**
- Consumes: everything from Tasks 0–7, plus `build_daily_snapshots`/`normalize_price_snapshots` (Task 0's modified version), `combine_edges` from `src.data.graph_edges`, `build_sector_graph`/`edge_index_for_tickers` from `src.models.graph_model`, `set_seed` from `src.utils.seed`.
- Produces: `results/experiment7_macro_conditioned/results.csv` (one row per completed run) and one `metrics.json` per run.

Training reuses Experiment 4's fixed pattern verbatim — gradient accumulation over 16 days, shuffled day order, early stopping on validation MCC — because that fix is what made graph training work at all.

### Open conflict: which set drives early stopping

The spec says *"Model selection must not consume the protocol dev set for tuning. Selection folds are carved from train."* The runner below early-stops on the **protocol dev set**, contradicting that line. This is deliberate and must be resolved by the user before the full ladder runs, not silently:

- **Early-stop on protocol dev (what the runner does).** Matches Experiments 1/3/4/5/6 exactly, so ladder row 1 is directly comparable to Experiment 6's price-only control — which is the entire reason row 1 exists. Cost: inherits the 42-day dev set's noise, a plausible driver of Experiment 6's 0.0247 seed spread.
- **Early-stop on a train-carved fold (what the spec asks for).** Lower selection noise, protocol dev stays untouched. Cost: row 1 is no longer comparable to Experiment 6's number, breaking the chain the ladder is anchored to.

**Recommendation: keep protocol dev**, because comparability is what makes the ladder interpretable, and the 5-seed requirement already exists to absorb the selection noise. If the user prefers the spec's version, change `run_epoch(model, dev_snaps, ctx)` to use a fold carved from the last 42 training days and note the broken comparison in the write-up.

- [ ] **Step 1: Write the embedding cache script**

Create `scripts/cache_experiment7_embeddings.py`:

```python
"""Embeds every text Experiment 7 needs, once, into a disk cache.

FinBERT on CPU is the experiment's single largest fixed cost, and the 40-run
ladder must not pay it 40 times. Run this after scripts/fetch_fnspid.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.micro_encoder import load_ticker_tweets
from src.models.finbert_cache import EmbeddingCache, cap_per_day, embed_texts

CACHE_PATH = "data/processed/experiment7_embeddings.pt"
CAP = 64


def main() -> None:
    cache = EmbeddingCache.load(CACHE_PATH)
    print(f"existing cache: {len(cache)} vectors")

    texts = set()
    for name in ("macro", "micro"):
        path = f"data/processed/fnspid_{name}.parquet"
        if not os.path.exists(path):
            print(f"MISSING {path} - run scripts/fetch_fnspid.py first")
            sys.exit(1)
        rows = pd.read_parquet(path).to_dict("records")
        capped = cap_per_day(rows, n=CAP)
        print(f"{name}: {len(rows):,} rows -> {len(capped):,} after per-day cap {CAP}")
        texts.update(r["headline"] for r in capped)

    tweets = load_ticker_tweets("dataset/final/stocknet_final_modeling_set.parquet")
    for segments in tweets.values():
        texts.update(segments[-CAP:])
    print(f"unique texts to embed: {len(texts):,}")

    pending = [t for t in sorted(texts) if cache.get(t) is None]
    print(f"already cached: {len(texts) - len(pending):,} | to embed: {len(pending):,}")

    batch = 256
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        for text, vector in zip(chunk, embed_texts(chunk)):
            cache.put(text, vector)
        cache.save(CACHE_PATH)          # checkpoint every batch; safe to resume
        print(f"  {min(start + batch, len(pending)):,}/{len(pending):,}", flush=True)

    print(f"done. cache holds {len(cache)} vectors at {CACHE_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the embedding cache**

```bash
python -u scripts/cache_experiment7_embeddings.py
```

Expected: prints per-day cap effects and finishes with a cache summary. Resumable — re-running skips already-embedded text.

- [ ] **Step 3: Write the ladder runner**

Create `scripts/run_experiment7_ladder.py`:

```python
"""Runs Experiment 7's 8-row ablation ladder at 5 seeds (40 runs).

Results are appended after EVERY run and completed cells are skipped on
restart, so an interrupted session resumes where it stopped.

Every row reports pooled AND within-day metrics: 48.3% of test label variance
is between-day, so pooled MCC alone cannot distinguish "discriminates stocks"
from "predicts market direction".
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn

from src.data.graph_edges import combine_edges
from src.data.graph_dataset import build_daily_snapshots, normalize_price_snapshots
from src.data.macro_encoder import build_macro_vectors, load_macro_calendar
from src.data.micro_encoder import build_micro_embeddings, load_ticker_tweets
from src.data.tweet_graph import build_tweet_edges
from src.models.finbert_cache import EmbeddingCache
from src.models.graph_model import build_sector_graph, edge_index_for_tickers
from src.models.macro_conditioned import MacroConditionedGraphNet
from src.utils.metrics import pooled_metrics, within_day_metrics
from src.utils.seed import set_seed

EXACT = "dataset/stocknet_protocol_a_exact.parquet"
FINAL = "dataset/final/stocknet_final_modeling_set.parquet"
SECTOR_CSV = "dataset/final/sector_mapping.csv"
CALENDAR = "dataset/macro_calendar.csv"
CACHE_PATH = "data/processed/experiment7_embeddings.pt"
OUT_DIR = "results/experiment7_macro_conditioned"
RESULTS_CSV = os.path.join(OUT_DIR, "results.csv")

PATIENCE, MAX_EPOCHS, LR, GRAD_ACCUM_DAYS = 10, 100, 1e-3, 16
WINSORIZE_LIMIT = 10.0
SEEDS = [0, 1, 2, 3, 4]

# (name, kwargs, graph_kind, shuffle_macro)
LADDER = [
    ("0_macro_only",      dict(macro_only=True),                              "none",   False),
    ("1_price_only",      dict(use_micro=False, use_graph=False, conditioning="none"), "none",   False),
    ("2_micro_no_graph",  dict(use_micro=True,  use_graph=False, conditioning="none"), "none",   False),
    ("3_graph_no_cond",   dict(use_micro=True,  use_graph=True,  conditioning="none"), "tweet",  False),
    ("4_full",            dict(use_micro=True,  use_graph=True,  conditioning="micro"), "tweet", False),
    ("5_shuffled_macro",  dict(use_micro=True,  use_graph=True,  conditioning="micro"), "tweet", True),
    ("6_attention_site",  dict(use_micro=True,  use_graph=True,  conditioning="attention"), "tweet", False),
    ("7_sector_graph",    dict(use_micro=True,  use_graph=True,  conditioning="micro"), "sector", False),
]


def build_everything():
    df = pd.read_parquet(EXACT)
    tickers = sorted(df["Ticker"].unique())
    days = sorted(pd.to_datetime(df["Target_Date"]).dt.strftime("%Y-%m-%d").unique())

    cache = EmbeddingCache.load(CACHE_PATH)
    if len(cache) == 0:
        print("empty embedding cache - run scripts/cache_experiment7_embeddings.py")
        sys.exit(1)

    macro_rows = pd.read_parquet("data/processed/fnspid_macro.parquet").to_dict("records")
    micro_rows = pd.read_parquet("data/processed/fnspid_micro.parquet").to_dict("records")
    calendar = load_macro_calendar(CALENDAR)
    tweets = load_ticker_tweets(FINAL)

    macro_vectors = build_macro_vectors(macro_rows, calendar, days, cache)
    micro_embeddings = build_micro_embeddings(tweets, micro_rows, tickers, days, cache)

    tweet_edges = {d: build_tweet_edges(micro_embeddings, tickers, d) for d in days}
    sector_graph = build_sector_graph(SECTOR_CSV)

    splits = tuple(build_daily_snapshots(df, s) for s in ("train", "dev", "test"))
    normalize_price_snapshots(*splits, winsorize_limit=WINSORIZE_LIMIT)
    return splits, macro_vectors, micro_embeddings, tweet_edges, sector_graph


def edges_for(kind, snap, tweet_edges, sector_graph):
    if kind == "none":
        return torch.empty((2, 0), dtype=torch.long)
    if kind == "sector":
        return edge_index_for_tickers(sector_graph, snap.tickers)
    # Reuses Experiment 6's tested combine_edges rather than reimplementing the
    # local-index/both-directions logic. The "sector" key is just relation id 0
    # from RELATION_IDS -- these are tweet edges, and the model is
    # relation-agnostic (plain GATConv), so the id carries no meaning here.
    edge_index, _ = combine_edges({"sector": tweet_edges.get(snap.date, set())},
                                  snap.tickers)
    return edge_index


def micro_tensor(snap, micro_embeddings):
    rows = []
    for ticker in snap.tickers:
        entry = micro_embeddings.get((ticker, snap.date))
        if entry is None:
            rows.append(torch.zeros(768))
        else:
            rows.append((entry[0] + entry[1]) / 2.0)
    return torch.stack(rows)


def run_epoch(model, snapshots, ctx, optimizer=None):
    macro_vectors, micro_embeddings, tweet_edges, sector_graph, kind, shuffled = ctx
    is_train = optimizer is not None
    model.train(is_train)
    order = list(snapshots)
    if is_train:
        random.shuffle(order)
        optimizer.zero_grad()

    probs_all, targets_all, days_all, total_loss = [], [], [], 0.0
    for i, snap in enumerate(order):
        macro = macro_vectors[shuffled[snap.date] if shuffled else snap.date]
        logits = model(snap.price_seq, snap.price_mask,
                       edges_for(kind, snap, tweet_edges, sector_graph),
                       micro_tensor(snap, micro_embeddings), macro)
        loss = nn.functional.cross_entropy(logits, snap.target)
        if is_train:
            (loss / GRAD_ACCUM_DAYS).backward()
            if (i + 1) % GRAD_ACCUM_DAYS == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * len(snap.tickers)
        probs_all.append(torch.softmax(logits, dim=-1)[:, 1].detach())
        targets_all.append(snap.target)
        days_all.extend([snap.date] * len(snap.tickers))

    probs = torch.cat(probs_all).numpy()
    targets = torch.cat(targets_all).numpy()
    metrics = pooled_metrics(targets, probs)
    metrics.update(within_day_metrics(targets, probs, days_all))
    return total_loss / len(targets), metrics


def train_one(row_name, kwargs, kind, shuffle, seed, splits, shared):
    set_seed(seed)
    macro_vectors, micro_embeddings, tweet_edges, sector_graph = shared
    train_snaps, dev_snaps, test_snaps = splits

    shuffled = None
    if shuffle:
        days = sorted(macro_vectors)
        permuted = list(days)
        random.Random(seed).shuffle(permuted)
        shuffled = dict(zip(days, permuted))

    ctx = (macro_vectors, micro_embeddings, tweet_edges, sector_graph, kind, shuffled)
    model = MacroConditionedGraphNet(**kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_mcc, best_epoch, best_state, stale = -1.0, -1, None, 0
    for epoch in range(MAX_EPOCHS):
        run_epoch(model, train_snaps, ctx, optimizer)
        _, val = run_epoch(model, dev_snaps, ctx)
        if val["mcc"] > best_mcc:
            best_mcc, best_epoch, stale = val["mcc"], epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    _, test = run_epoch(model, test_snaps, ctx)
    test["n_samples"] = sum(len(s.tickers) for s in test_snaps)
    test["best_epoch"] = best_epoch
    test["best_val_mcc"] = best_mcc
    return test


def already_done(row_name, seed):
    if not os.path.exists(RESULTS_CSV):
        return False
    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return False
    if "row" not in df.columns or "seed" not in df.columns:
        return False
    return bool(((df["row"] == row_name) & (df["seed"] == seed)).any())


def record(row_name, seed, test):
    run_dir = os.path.join(OUT_DIR, f"{row_name}__seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump({"test": {"label": f"{row_name} seed={seed}", "backend": None, **test}},
                  fh, indent=2)
    row = {"row": row_name, "seed": seed, **test}
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a",
                               header=not os.path.exists(RESULTS_CSV), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=None,
                    help="row names to run; default all")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("building embeddings, vectors and graphs...", flush=True)
    splits, macro_vectors, micro_embeddings, tweet_edges, sector_graph = build_everything()
    shared = (macro_vectors, micro_embeddings, tweet_edges, sector_graph)
    density = [len(v) for v in tweet_edges.values()]
    print(f"tweet graph: mean {sum(density)/max(len(density),1):.1f} edges/day, "
          f"max {max(density) if density else 0}")

    for row_name, kwargs, kind, shuffle in LADDER:
        if args.rows and row_name not in args.rows:
            continue
        for seed in SEEDS:
            if already_done(row_name, seed):
                print(f"  skip  {row_name} seed={seed}")
                continue
            print(f"  train {row_name} seed={seed} ...", flush=True)
            test = train_one(row_name, kwargs, kind, shuffle, seed, splits, shared)
            record(row_name, seed, test)
            print(f"        pooled MCC={test['mcc']:+.4f} AUC={test['auc']:.4f} | "
                  f"within-day MCC={test['within_day_mcc']:+.4f} "
                  f"({test['days_scored']} days)")
            if test["mcc"] > 0.15:
                print("        *** TRIPWIRE: pooled MCC > 0.15 - audit for leakage "
                      "before treating this as a result ***")

    print(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-test one cheap row before committing to all 40 runs**

```bash
python -u scripts/run_experiment7_ladder.py --rows 0_macro_only 2>&1 | tail -20
```

Expected: 5 seeds of the macro-only row complete, `n_samples` is 3720, and within-day `days_scored` is reported. **The macro-only row's within-day MCC should be ~0** by construction — every ticker on a day gets an identical prediction. If it is not, stop: the within-day metric or the model's `macro_only` path is wrong.

- [ ] **Step 5: Run the full ladder**

```bash
nohup python -u scripts/run_experiment7_ladder.py > results/experiment7_macro_conditioned/ladder.log 2>&1 &
```

40 runs; expect roughly a day of wall-clock on CPU based on Experiment 6's timings. Resumable.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment7_ladder.py scripts/cache_experiment7_embeddings.py
git commit -m "Add Experiment 7 ablation ladder runner and embedding cache script"
```

---

## Task 9: Figures and explainability readout

**Files:**
- Create: `scripts/make_experiment7_figures.py`

**Interfaces:**
- Consumes: `results/experiment7_macro_conditioned/results.csv` (Task 8), the embedding cache, `load_macro_calendar` (Task 3), `MacroConditionedGraphNet.last_film` (Task 6).
- Produces: PNGs in `results/experiment7_macro_conditioned/figures/`.

- [ ] **Step 1: Write the figures script**

Create `scripts/make_experiment7_figures.py`:

```python
"""Experiment 7 figures: the ablation ladder, pooled vs within-day, and the
FiLM explainability readout.

The gamma/beta readout is a deliverable independent of the MCC outcome -- if
the ladder shows no gain, "macro events measurably change how per-ticker text
is interpreted" is still a reportable finding.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.macro_encoder import load_macro_calendar

OUT_DIR = "results/experiment7_macro_conditioned"
FIG_DIR = os.path.join(OUT_DIR, "figures")
REFERENCES = {
    "Exp1 LSTM+tweets": 0.0922,
    "Exp6 best (RGCN sec+corr)": 0.0677,
    "Exp6 GAT sector (3-seed)": 0.0160,
}


def ladder_figure(df: pd.DataFrame) -> None:
    agg = df.groupby("row").agg(
        pooled=("mcc", "mean"), pooled_std=("mcc", "std"),
        within=("within_day_mcc", "mean"), within_std=("within_day_mcc", "std"),
    ).reset_index().sort_values("row")

    x = range(len(agg))
    width = 0.38
    plt.figure(figsize=(13, 6))
    plt.bar([i - width / 2 for i in x], agg["pooled"], width,
            yerr=agg["pooled_std"].fillna(0), capsize=3, label="pooled MCC")
    plt.bar([i + width / 2 for i in x], agg["within"], width,
            yerr=agg["within_std"].fillna(0), capsize=3, label="within-day MCC")
    for name, value in REFERENCES.items():
        plt.axhline(value, linestyle="--", linewidth=1, alpha=0.7)
        plt.text(len(agg) - 0.4, value, f" {name} ({value:+.4f})",
                 va="bottom", fontsize=7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(list(x), agg["row"], rotation=25, ha="right", fontsize=8)
    plt.ylabel("test MCC (mean over 5 seeds)")
    plt.title("Experiment 7 ablation ladder: pooled vs within-day MCC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_ladder.png"), dpi=150)
    plt.close()


def seed_spread_figure(df: pd.DataFrame) -> None:
    spread = df.groupby("row")["mcc"].apply(lambda g: g.max() - g.min())
    plt.figure(figsize=(11, 4))
    plt.bar(range(len(spread)), spread.values)
    plt.axhline(0.0247, linestyle="--", color="red", linewidth=1)
    plt.text(0, 0.0247, " Exp6 noise floor (0.0247)", va="bottom",
             fontsize=8, color="red")
    plt.xticks(range(len(spread)), spread.index, rotation=25, ha="right", fontsize=8)
    plt.ylabel("max - min test MCC across seeds")
    plt.title("Seed spread per ladder row (differences below this are noise)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "seed_spread.png"), dpi=150)
    plt.close()


def film_figure(film_csv: str) -> None:
    """Plots gamma/beta magnitude per trading day against macro release dates.

    Expects `film_csv` with columns date,gamma_norm,beta_norm -- written by the
    runner's explainability dump (see Step 2).
    """
    if not os.path.exists(film_csv):
        print("no film_by_day.csv yet; skipping the explainability figure")
        return
    film = pd.read_csv(film_csv).sort_values("date")
    calendar = load_macro_calendar("dataset/macro_calendar.csv")

    plt.figure(figsize=(13, 4.5))
    plt.plot(range(len(film)), film["gamma_norm"], linewidth=1, label="||gamma||")
    plt.plot(range(len(film)), film["beta_norm"], linewidth=1, label="||beta||")
    for i, day in enumerate(film["date"]):
        flags = calendar.get(str(day)[:10], (0, 0, 0))
        if flags[0]:
            plt.axvline(i, color="red", alpha=0.25, linewidth=1)
    plt.xlabel("trading day index (red = FOMC)")
    plt.ylabel("FiLM parameter norm")
    plt.title("Macro conditioning strength per trading day, against FOMC dates")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "film_by_day.png"), dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    results = os.path.join(OUT_DIR, "results.csv")
    if not os.path.exists(results):
        print("no results.csv yet; run scripts/run_experiment7_ladder.py first")
        return
    df = pd.read_csv(results)
    ladder_figure(df)
    seed_spread_figure(df)
    film_figure(os.path.join(OUT_DIR, "film_by_day.csv"))
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the FiLM dump to the runner**

Append to `scripts/run_experiment7_ladder.py`, and call it from `main()` after the ladder completes, for the best seed of row `4_full`:

```python
def dump_film_by_day(model, snapshots, ctx) -> None:
    """Records ||gamma|| and ||beta|| per trading day for the explainability
    figure. Uses the trained full model on the test split."""
    macro_vectors, micro_embeddings, tweet_edges, sector_graph, kind, shuffled = ctx
    model.eval()
    rows = []
    with torch.no_grad():
        for snap in snapshots:
            model(snap.price_seq, snap.price_mask,
                  edges_for(kind, snap, tweet_edges, sector_graph),
                  micro_tensor(snap, micro_embeddings), macro_vectors[snap.date])
            if model.last_film is None:
                continue
            gamma, beta = model.last_film
            rows.append({"date": snap.date,
                         "gamma_norm": float(gamma.norm()),
                         "beta_norm": float(beta.norm())})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "film_by_day.csv"), index=False)
```

- [ ] **Step 3: Verify the figures render**

Run: `python scripts/make_experiment7_figures.py`
Expected: writes `ablation_ladder.png` and `seed_spread.png`; prints the skip message for the FiLM figure if `film_by_day.csv` is not present yet.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_experiment7_figures.py scripts/run_experiment7_ladder.py
git commit -m "Add Experiment 7 figures and FiLM explainability readout"
```

---

## Task 10: Results write-up

**Files:**
- Create: `results/experiment7_macro_conditioned/EXPERIMENT7_MACRO_CONDITIONED.md`

**Blocked until `results.csv` holds all 40 runs.** An implementer reaching this task with an incomplete ladder should stop and report that, rather than writing a document with placeholder or invented numbers.

- [ ] **Step 1: Write the results document**

Follow `results/experiment6_dense_graph/EXPERIMENT6_DENSE_GRAPH.md`'s structure. Fill every number from the real `results.csv` — invent nothing. Required sections:

1. **Objective** — whether a tweet-learned graph whose micro signal is macro-conditioned beats the hand-designed graphs Experiment 6 measured.
2. **The MAN-SF non-reproduction caveat**, carried forward: no graph here is their Wikidata company-relation graph, and no result may be described as reproducing or beating 0.117 or 0.195.
3. **Task 0's finding** — the `Price3` fix, its measured effect on the re-run Experiment 6 cell, and explicitly whether Experiment 6's conclusions are affected.
4. **The between-day/within-day problem** — the measured variance decomposition (test 48.3% between-day, 32/64 days >80% one-directional, oracle pooled MCC +0.6248 / accuracy 0.8116) and why every row is reported twice.
5. **Data** — FNSPID coverage actually measured by Task 1 (not the spec's estimates), the per-day cap chosen and why, the English-filter reject rate, and tweet-graph density.
6. **Ablation ladder table** — all 8 rows, pooled and within-day, mean ± std across 5 seeds, against Experiment 1 (+0.0922), Experiment 6 best (+0.0677 ± 0.0255), Experiment 6 GAT sector (+0.0160 ± 0.0161), both constant predictors, and the market-direction oracle.
7. **Honest reading** — lead with within-day MCC and AUC. State plainly whether row 4 beats row 3 (conditioning), whether row 4 beats row 5 (events vs capacity), and whether row 0 explains row 4 (market direction vs stock discrimination). Any difference smaller than seed spread is called indistinguishable.
8. **Explainability readout** — what the FiLM parameters did around macro events, reported whether or not the MCC result was positive.
9. **Leakage audit** — that the tripwire was checked and what it found.
10. **Disclosure** — the per-day cap's effect on macro fidelity, the 17 uncovered join samples, epoch budget, and the shared-`gamma` limitation on intra-day reordering.
11. **How we should move forward.**

- [ ] **Step 2: Commit**

```bash
git add results/experiment7_macro_conditioned/
git commit -m "Add Experiment 7 results write-up"
```

---

## Deferred

**Cross-day recurrence (Experiment 8).** A GRU/LSTM over the sequence of daily graph embeddings per ticker. Every trading day is currently modeled independently with no memory across days. Deferred so Experiment 7 stacks one architectural novelty rather than two; if the conditioned graph works, this is the natural next step.

**Wikidata company-relation graph.** Feng et al. published relation encodings; valuable as a control anchoring the results table against a graph independently known to carry signal. Tracked separately so it does not entangle with this experiment's novelty.
