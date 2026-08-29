# Experiment 1 Summary — Exact StockNet Protocol Reproduction

*Full technical write-up: `results/experiment1_stocknet_exact/EXPERIMENT1_STOCKNET_EXACT.md`*

## What we did

Reproduced the original StockNet paper's (Xu & Cohen, ACL 2018) data
methodology — not their neural architecture, just how they built the
dataset — and evaluated it with three simple baseline models (Logistic
Regression, LSTM, MLP) across four feature sets (Price / +Fundamentals /
+Tweets / Full).

## How we did it

We didn't guess at StockNet's methodology from memory or from our own
project's existing conventions — we pulled the actual
[`stocknet-code`](https://github.com/yumoxu/stocknet-code) repository and
ported its real preprocessing logic (`DataPipe.py`, `config.yml`) line for
line:

- **Exact 88-ticker universe and official train/dev/test split dates**
  (`2014-01-01→2015-08-01→2015-10-01→2016-01-01`), copied verbatim from the
  official config, not our project's ad hoc split.
- **Calendar-anchored 5-day window** with variable real-trading-day length
  (weekends/holidays shrink it), zero-padded and masked rather than
  discarded — matching the official code's actual behavior instead of our
  first (incorrect) assumption that "5-day window" meant exactly 5 trading
  days always.
- **Buffer-zone filtering applied only to the target day** being predicted
  (±0.5%/0.55% movement thresholds), never to input days — a subtle but
  important distinction the official code enforces that our project's
  earlier preprocessing had gotten backwards.
- **Tweet-to-trading-day alignment**: every calendar day's tweets attach to
  the first actual trading day strictly after it, never the same day.
- Fundamentals were added on top as a project-specific extension (not part
  of real StockNet), re-aligned to each ticker's full trading calendar using
  the SEC EDGAR data already cached from earlier preprocessing work.

**Validation, not assumption**: the resulting dataset produced 26,619 total
samples with a 50.2%/49.8% up/down balance — matching the published
StockNet dataset's known scale (~26,614 samples) almost exactly. That
congruence is the evidence this reproduction is faithful, not just
plausible-looking.

## Why we did it this way

The project's prior baselines (Phase 1, Phase 1 v2) used this project's own
preprocessing conventions, which meant results couldn't be honestly compared
against the literature — there was no guarantee the split, labels, or
windowing matched what published numbers were actually measured on. Before
building anything more sophisticated (attention, fusion, graphs), we needed
one trustworthy number computed under conditions that are actually
comparable to MAN-SF's own ablation table. Otherwise every subsequent
"beats baseline" claim would be comparing apples to a different fruit.

## What we found

Best result: **LSTM on Price+Tweets (raw tweet count), MCC +0.092, F1
0.636**. This lands in a sensible range against MAN-SF's own reported
ablation (LSTM+price ≈0.002, GRU+social text ≈0.077) — close enough to trust
as a real baseline, not an artifact of a broken pipeline. Fundamentals alone
also beat price-only, but combining fundamentals with tweets didn't beat
tweets alone — the two signals didn't compound under simple concatenation.

## How we should move forward

This MCC +0.092 (LSTM, raw tweet count) is the number every subsequent
experiment needs to beat to justify its added complexity. It became the
baseline against which Experiments 2 and 3 were judged — and, as of this
writing, neither has beaten it yet (see their own summaries). Any future
architecture change should be evaluated against this same exact-protocol
dataset and split, not a different one, to keep comparisons honest.
