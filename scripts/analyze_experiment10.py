"""Experiment 10 analysis: does the graph make a difference, and how?

Three questions, each with its own contrast:

  Q1  Does MESSAGE PASSING help?        graph=True vs graph=False, same variant
  Q2  Does the graph's INFORMATION help? +peer vs plain, both with graph=False
  Q3  Do the architectural fixes help?   contrast / xsect vs plain, graph=True

Q2 is the one that matters most. Message passing averages neighbours, which is
the wrong operation for a metric that scores how stocks differ from each other.
Peer features carry the same neighbourhood information without that averaging,
so if Q2 is positive while Q1 is null, the edges hold real signal and the GNN
was simply the wrong way to read it.

Primary metric is within-day MCC throughout: pooled MCC on this benchmark is
roughly half market direction, and a day-direction oracle scores +0.6248 with no
stock-specific skill at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy import stats

RESULTS = "results/experiment10_multimodal/results.csv"
M = "within_day_mcc"


def cell(d, target, variant, use_graph, conv="gat"):
    """One conv only. The screening pass ran four convs all at seed 0, so mixing
    convs would treat four architectures as four seeds of one model."""
    return d[(d.target == target) & (d.variant == variant) &
             (d.use_graph == use_graph) & (d.conv == conv)].sort_values("seed")


def contrast(a, b, label, metric=M):
    """Reports a difference only when both sides have matching seed sets."""
    shared = sorted(set(a.seed) & set(b.seed))
    if len(shared) < 2 or len(shared) < min(len(a), len(b)):
        print(f"  {label:<44} INCOMPLETE ({len(a)} vs {len(b)} seeds)")
        return
    xa = a[a.seed.isin(shared)].sort_values("seed")[metric].values
    xb = b[b.seed.isin(shared)].sort_values("seed")[metric].values
    p = stats.ttest_ind(xa, xb, equal_var=False).pvalue
    spread = max(xa.max() - xa.min(), xb.max() - xb.min())
    flag = "exceeds noise" if abs(xa.mean() - xb.mean()) > spread else "below noise"
    print(f"  {label:<44} {xa.mean():+.4f} vs {xb.mean():+.4f}  "
          f"diff {xa.mean()-xb.mean():+.4f}  p={p:.4f}  n={len(shared)}  [{flag}]")


def main() -> None:
    if not os.path.exists(RESULTS):
        sys.exit(f"missing {RESULTS}")
    d = pd.read_csv(RESULTS)
    if "variant" not in d.columns:
        d["variant"] = "plain"
    d["variant"] = d["variant"].fillna("plain")

    print("=== cells available (all convs) ===")
    summary = d.groupby(["target", "variant", "conv", "use_graph"]).agg(
        within=(M, "mean"), spread=(M, lambda s: s.max() - s.min()),
        pooled=("mcc", "mean"), acc=("accuracy", "mean"), n=(M, "count"))
    print(summary.round(4).to_string())

    for target in sorted(d.target.unique()):
        print(f"\n=== {target.upper()} ===")
        print(" Q1 does message passing help?")
        for variant in sorted(d[d.target == target].variant.unique()):
            contrast(cell(d, target, variant, True), cell(d, target, variant, False),
                     f"  [{variant}] graph vs no-graph")

        print(" Q2 does the graph's INFORMATION help, without message passing?")
        contrast(cell(d, target, "plain+peer", False), cell(d, target, "plain", False),
                 "  peer features vs none (both no-graph)")

        print(" Q3 do the architectural fixes rescue message passing?")
        for variant in ("contrast", "xsect", "contrast+xs"):
            if len(cell(d, target, variant, True)):
                contrast(cell(d, target, variant, True), cell(d, target, "plain", True),
                         f"  [{variant}] vs plain (both graph)")

        print(" best cell on this target:")
        best = d[(d.target == target) & (d.conv == "gat")].groupby(
            ["variant", "use_graph"])[M].mean()
        if len(best):
            (v, g), score = best.idxmax(), best.max()
            print(f"    {v}  graph={g}  within-day {score:+.4f}")

    print(f"\nn_samples: {sorted(d.n_samples.unique())}  |  total cells: {len(d)}")


if __name__ == "__main__":
    main()
