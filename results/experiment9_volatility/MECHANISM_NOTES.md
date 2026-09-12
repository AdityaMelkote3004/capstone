# Volatility result — mechanism checks

Everything here was run to answer one question: is the +0.1433 volatility MCC a
real effect, an artefact, or a leak? Summary: **real and leak-free, but its
economic interpretation is not yet established.**

## Ruled out

**Leakage.** Max |correlation| between any of the 12 input price features and the
target day's move is **0.030**. A feature containing the target would be ~1.0.

**A calendar / weekend artefact.** Post-weekend days carry more accumulated news,
so a model could in principle score by detecting short windows. It does not:

| Features | Test MCC | Accuracy |
|---|---|---|
| `Window_Length` alone | +0.0340 | **0.5005** |
| 12 price features | +0.1072 | 0.5855 |
| price + `Window_Length` | +0.1073 | 0.5860 |

Window length alone predicts at chance accuracy, and adds **+0.0001** on top of
price. Mean \|move\| by window length is nearly flat (0.01450 → 0.01502).

**An LSTM artefact.** Plain logistic regression on the same 12 features reaches
MCC **+0.1072**, recovering most of the neural model's +0.1433. The signal is in
the features, not the architecture.

## Two corrections to earlier analysis in this repo

1. `_window_mask` fills from the **start**, so short windows pad at the **end**.
   `Day4` is zero for many rows. An earlier reading of the Day1→Day4 correlation
   decline as *temporal decay* was wrong — it is largely **padding dilution**.
   Restricting to `Window_Length == 4` flattens it (−0.122 / −0.113 / −0.112 /
   −0.097).
2. `Price1` has mean **+0.090**, not ~0, so it is not a return. An earlier
   "volatility clustering" check that correlated `|Price1|` against `|move|` was
   therefore meaningless — taking the absolute value of an already-positive
   quantity tests nothing.

## Unresolved

The per-feature correlation with \|next-day move\| is consistently **negative**
(−0.10 to −0.14), and stays negative on full-length windows only. Textbook
volatility clustering would be positive.

We cannot currently say what `Price1`/`Price2` represent. They are StockNet's
`data[3:6]` from `price/preprocessed/{TICKER}.txt`, and **those raw files are not
present in this checkout**, so the definition cannot be confirmed from source.
Their shape (mean +0.09/+0.07, mostly positive, shrinking toward 0 from `Day1` to
`Day4`) is *consistent with* prices normalised against the most recent close in
the window, which would make them trend rather than spread measures — but that is
a hypothesis, not a verified fact, and it is not asserted as the mechanism.

`Price3` is the known-degenerate feature (sd **199.9**, range ±6495, driven by
BRK-A and PCLN); it is winsorized at ±10 before use.

**Before publication:** obtain `stocknet-dataset/price/preprocessed/` and confirm
what `data[3:6]` are. The result stands without this; the *explanation* does not.
