"""Does the graph advantage survive with the news half removed?

Micro node features concatenate a tweet vector (strictly <= t-1) with a news
vector windowed to day t's 09:30 open. The close-to-close label already contains
the overnight gap, so the news half is partly CONTEMPORANEOUS with the label.
The +0.0425 graph effect therefore has two candidate explanations:

  (a) message passing over text features carries predictive signal
  (b) message passing propagates overnight-gap information across neighbours

Rows suffixed '__tweetonly' were trained with the news half zeroed. If the graph
advantage holds there, (a) survives and (b) is ruled out as its sole cause.
Price-only (row 1) has no micro input at all, so it is the shared baseline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy import stats

RESULTS = "results/experiment7_macro_conditioned/results.csv"
M = "within_day_mcc"
ALPHA_BONF = 0.05 / 7   # same declared family as the main ladder
PAIRS = [
    ("2_micro_no_graph", "micro features, no graph"),
    ("3_graph_no_cond", "tweet graph, no conditioning"),
    ("8_sector_no_cond", "sector graph, no conditioning"),
]


def summarise(d: pd.DataFrame, row: str):
    s = d[d.row == row]
    return (s[M].mean(), s[M].std(), s["auc"].mean(), len(s)) if len(s) else None


def main() -> None:
    d = pd.read_csv(RESULTS)
    base = d[d.row == "1_price_only"][M]
    ladder = d[~d.row.str.endswith("__tweetonly")]
    noise = ladder.groupby("row")[M].agg(lambda x: x.max() - x.min()).mean()
    print(f"baseline 1_price_only: {base.mean():+.4f} (n={len(base)})\n")

    hdr = f"{'row':<32}{'with news':>12}{'tweet only':>12}{'delta':>9}{'n':>5}"
    print(hdr); print("-" * len(hdr))
    for row, label in PAIRS:
        full, only = summarise(d, row), summarise(d, row + "__tweetonly")
        if not full:
            continue
        if not only:
            print(f"{label:<32}{full[0]:>+12.4f}{'pending':>12}{'':>9}{full[3]:>5}")
            continue
        print(f"{label:<32}{full[0]:>+12.4f}{only[0]:>+12.4f}"
              f"{only[0]-full[0]:>+9.4f}{only[3]:>5}")

    print("\n=== the decisive contrast: graph vs price-only, news removed ===")
    for row, label in PAIRS:
        if row == "2_micro_no_graph":
            continue
        s = d[d.row == row + "__tweetonly"][M]
        if len(s) < 2:
            print(f"  {label:<32} pending ({len(s)} seeds)")
            continue
        diff = s.mean() - base.mean()
        p = stats.ttest_ind(s.values, base.values, equal_var=False).pvalue
        # Held to the same bar as the main ladder, not merely p<0.05: the effect
        # must also clear the seed-spread noise floor and the Bonferroni alpha.
        # A nominally significant effect smaller than the noise is not a result.
        flags = []
        if p < 0.05 and diff > 0:
            flags.append("nominally significant")
        if diff > noise:
            flags.append("exceeds noise floor")
        else:
            flags.append(f"BELOW noise floor ({noise:.4f})")
        if p < ALPHA_BONF:
            flags.append("survives Bonferroni")
        else:
            flags.append(f"fails Bonferroni (a={ALPHA_BONF:.4f})")
        print(f"  {label:<32} {diff:>+8.4f}  p={p:.4f}  {sum(s > 0)}/{len(s)} seeds +")
        print(f"  {'':<32} {' | '.join(flags)}")

    print("\nn_samples across all cells:", sorted(d["n_samples"].unique()))
    print("max pooled MCC (tripwire 0.15):", f"{d['mcc'].max():+.4f}")


if __name__ == "__main__":
    main()
