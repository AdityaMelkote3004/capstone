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
