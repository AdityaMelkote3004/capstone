"""Turns results.csv into the comparison table the write-up quotes.

Every number in EXPERIMENT7_MACRO_CONDITIONED.md that is not a raw per-cell
metric comes from here, so the write-up can be regenerated from the CSV.

The Bonferroni family below is declared POST HOC -- the spec did not
pre-register a correction. It is the set of contrasts the ladder was built to
make, fixed before any p-value in it was read, but a reader should treat it as
author-chosen. n=5 seeds is a small sample for a t-test; the seed spread is
reported alongside every contrast as the honest noise reference.
"""

import argparse
import itertools
import os
import sys

import pandas as pd
from scipy import stats

RESULTS = "results/experiment7_macro_conditioned/results.csv"

# (label, row_a, row_b, what a positive difference would mean)
CONTRASTS = [
    ("graph vs price-only",     "3_graph_no_cond", "1_price_only",     "the tweet graph adds signal"),
    ("micro vs price-only",     "2_micro_no_graph", "1_price_only",    "text alone adds signal"),
    ("graph vs micro-no-graph", "3_graph_no_cond", "2_micro_no_graph", "graph structure adds beyond text"),
    ("conditioning vs shuffled","4_full",          "5_shuffled_macro", "macro conditioning is real, not noise"),
    ("conditioning vs none",    "4_full",          "3_graph_no_cond",  "conditioning helps over the plain graph"),
    ("attention site vs FiLM",  "6_attention_site","3_graph_no_cond",  "conditioning at attention helps"),
    ("tweet graph vs sector",   "3_graph_no_cond", "8_sector_no_cond", "the LEARNED graph beats sector cliques"),
]
ALPHA = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="within_day_mcc")
    args = ap.parse_args()

    if not os.path.exists(RESULTS):
        sys.exit(f"missing {RESULTS}")
    d = pd.read_csv(RESULTS)
    m = args.metric

    per_row = d.groupby("row").agg(
        mean=(m, "mean"), sd=(m, "std"),
        spread=(m, lambda s: s.max() - s.min()),
        pooled_mcc=("mcc", "mean"), auc=("auc", "mean"),
        epoch=("best_epoch", "mean"), seeds=(m, "count"))
    print(f"=== per-row ({m}) ===")
    print(per_row.round(4).to_string())

    noise = per_row["spread"].mean()
    print(f"\nnoise floor (mean seed spread): {noise:+.4f}")
    print(f"tripwire, max pooled MCC:       {d['mcc'].max():+.4f}  (fires at 0.15)")
    print(f"n_samples across all cells:     {sorted(d['n_samples'].unique())}")

    present = [c for c in CONTRASTS if c[1] in per_row.index and c[2] in per_row.index]
    alpha_c = ALPHA / len(present)
    print(f"\n=== contrasts (Bonferroni over {len(present)} declared comparisons, "
          f"alpha={alpha_c:.4f}) ===")
    hdr = f"{'contrast':<26}{'diff':>9}{'unpaired p':>12}{'paired p':>10}{'>noise':>8}  {'survives':>8}"
    print(hdr); print("-" * len(hdr))
    for label, a, b, _meaning in present:
        sa = d[d.row == a].set_index("seed")[m]
        sb = d[d.row == b].set_index("seed")[m]
        # A half-finished row must not be quietly compared on unequal seeds --
        # that would be a real difference measured against a different sample.
        shared = sorted(set(sa.index) & set(sb.index))
        if len(shared) < len(sa) or len(shared) < len(sb):
            print(f"{label:<26}   INCOMPLETE ({a}: {len(sa)} seeds, {b}: {len(sb)} seeds)")
            continue
        xa, xb = sa.loc[shared].values, sb.loc[shared].values
        diff = xa.mean() - xb.mean()
        pu = stats.ttest_ind(xa, xb, equal_var=False).pvalue
        pp = stats.ttest_rel(xa, xb).pvalue
        print(f"{label:<26}{diff:>+9.4f}{pu:>12.4f}{pp:>10.4f}"
              f"{('yes' if abs(diff) > noise else 'no'):>8}"
              f"{('YES' if pu < alpha_c else 'no'):>10}")

    # AUC is threshold-free, so it cannot be moved by the within-day median
    # thresholding. If it disagrees with within-day MCC on a contrast, the
    # MCC result is an artefact of where the threshold fell.
    print("\n=== same contrasts on AUC (independent corroboration) ===")
    for label, a, b, _meaning in present:
        sa = d[d.row == a].set_index("seed")
        sb = d[d.row == b].set_index("seed")
        shared = sorted(set(sa.index) & set(sb.index))
        if len(shared) < len(sa) or len(shared) < len(sb):
            print(f"  {label:<26} INCOMPLETE")
            continue
        xa = sa.loc[shared, "auc"].values
        xb = sb.loc[shared, "auc"].values
        pu = stats.ttest_ind(xa, xb, equal_var=False).pvalue
        agrees = "agrees" if (xa.mean() - xb.mean()) * (
            sa.loc[shared, m].mean() - sb.loc[shared, m].mean()) > 0 else "DISAGREES"
        print(f"  {label:<26} {xa.mean():.4f} vs {xb.mean():.4f}  "
              f"diff {xa.mean()-xb.mean():+.4f}  p={pu:.4f}  [{agrees} in sign]")

    print("\n=== per-seed, key rows ===")
    for row in ["1_price_only", "3_graph_no_cond", "4_full", "5_shuffled_macro",
                "7_sector_graph", "8_sector_no_cond"]:
        if row not in per_row.index:
            continue
        v = d[d.row == row].sort_values("seed")[m].values
        print(f"  {row:<20} " + " ".join(f"{x:+.4f}" for x in v)
              + f"   ({sum(x > 0 for x in v)}/{len(v)} positive)")


if __name__ == "__main__":
    main()
