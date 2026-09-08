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
