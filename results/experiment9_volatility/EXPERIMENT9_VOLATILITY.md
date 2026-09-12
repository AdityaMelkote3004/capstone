# Experiment 9 — The limitation is the question, not the model

**Status:** Complete. 20 cells (4 rows × 5 seeds) on the volatility target,
compared against 65 direction-target cells from Experiment 8.
**Run:** `scripts/run_experiment8_lowdim_text.py --target volatility`
**Raw results:** `results/experiment8_lowdim_text/results_volatility.csv`
**Mechanism checks:** `MECHANISM_NOTES.md` (read this before citing the result)

---

## 1. The question

Experiments 1–8 tried seven architectures against next-day **direction** and
found essentially nothing. Rather than try an eighth architecture, Experiment 9
changes the **target** and holds everything else fixed: same 87 tickers, same
protocol-A splits, same LSTM, same features, same five seeds, same test set of
3,720 samples.

New label: **is tomorrow's absolute move larger than the training-set median?**
The threshold (0.0112) is computed on train rows only, so it cannot leak.

## 2. Result

| Row | DIRECTION pooled | DIRECTION within-day | **VOLATILITY pooled** | **VOLATILITY within-day** | Acc | AUC |
|---|---|---|---|---|---|---|
| price only | −0.0086 | +0.0088 | **+0.1433** | +0.1322 | 58.5% | 0.600 |
| + tweet counts | +0.0069 | +0.0262 | +0.1105 | +0.1122 | 57.7% | 0.586 |
| + counts + PCA-2 text | +0.0338 | +0.0272 | **+0.1493** | **+0.1472** | 58.8% | 0.604 |
| + counts + PCA-8 text | +0.0289 | +0.0178 | +0.1445 | +0.1434 | 59.0% | 0.604 |

**Best direction result in the entire project: +0.0338. Best volatility result:
+0.1493.** Price features alone, with no text and no graph, score +0.1433 —
roughly an order of magnitude above anything seven experiments extracted for
direction.

**Within-day MCC is +0.1322 to +0.1472.** This matters more than the pooled
figure. Within-day scores rank stocks *against each other on the same day*, so
the model is identifying **which stocks** will move most, not merely noticing
that the whole market is turbulent. The between-day confound — 48.3% of label
variance, documented in Experiment 7 §4 — is what hollowed out the direction
results. Volatility clears it.

### 2.1 Text still adds nothing

| vs price-only, volatility target | Pooled | p | Within-day | p |
|---|---|---|---|---|
| + tweet counts | −0.0328 | 0.295 | −0.0199 | 0.421 |
| + counts + PCA-2 | +0.0060 | 0.793 | +0.0150 | 0.454 |
| + counts + PCA-8 | +0.0012 | 0.958 | +0.0113 | 0.601 |

Adding FinBERT to a volatility model changes nothing (p = 0.96). This is the
same answer Experiments 1, 2, 3, 7 and 8 gave for direction, now confirmed on a
target where the model demonstrably *can* learn.

## 3. Why this is not an artefact

Three alternative explanations were tested and rejected. Full detail in
`MECHANISM_NOTES.md`.

**Leakage.** Max |correlation| between any of the 12 input features and the
target day's move: **0.030**. A feature containing the target would be ~1.0.

**A weekend / calendar effect.** Post-weekend days accumulate more news, so a
model could score by detecting short windows:

| Features | Test MCC | Accuracy |
|---|---|---|
| `Window_Length` alone | +0.0340 | **0.5005** |
| 12 price features | +0.1072 | 0.5855 |
| price + `Window_Length` | +0.1073 | 0.5860 |

Window length alone predicts at chance accuracy and contributes **+0.0001** on
top of price.

**An architecture artefact.** Plain **logistic regression** on the same 12
features reaches MCC **+0.1072**, recovering most of the neural model's +0.1433.
The signal is in the data, not the LSTM.

## 4. What is NOT established

**Why it works.** The per-feature correlation with |next-day move| is
consistently **negative** (−0.10 to −0.14), where textbook volatility clustering
predicts positive. `Price1`/`Price2` are StockNet's `data[3:6]`, and those raw
preprocessed files **are not present in this checkout**, so their definition
cannot be confirmed from source. A hypothesis (prices normalised against the
window's last close, making them trend rather than spread measures) is recorded
in `MECHANISM_NOTES.md` and explicitly **not** asserted as the mechanism.

The result is leak-free and replicated across 5 seeds. The *explanation* is open.
Obtain `stocknet-dataset/price/preprocessed/` before publication.

**Comparability to the direction numbers.** MCC across two different targets is
not a like-for-like comparison — the volatility label is 58% positive in test
versus 51% for direction. The correct reading is *"this target is predictable and
that one was not,"* never *"the model improved."*

**Benchmark claims.** None. Published StockNet/MAN-SF figures are direction
results; there is no volatility baseline here to compare against. A GARCH(1,1)
baseline is the obvious next control and was not run.

**The seed spread is large** (0.065–0.100 pooled). Every row is strongly
positive on all five seeds, but individual cell values should not be quoted to
four decimals.

## 5. What this means for the project

Three findings, which fit together into one claim:

1. **Next-day direction is not predictable** from this data. Seven experiments,
   five architectures — and in each case where a positive result appeared, the
   mechanism was identified: seed luck (Exp 1), feature/label timing overlap
   (Exp 7), missingness encoding (Exp 8).
2. **Next-day volatility is predictable** by the same pipeline, at +0.1433 MCC
   and 58.8% accuracy, including cross-sectionally within a day.
3. **Financial text adds nothing to either.** Every apparent text gain across
   eight experiments traced back to volume, coverage, or timing — never to
   semantic content.

The honest headline is not "our model beat the benchmark" and not "nothing
worked." It is: **the ceiling was set by the question being asked, and we can
demonstrate that by changing only the question.**

## 6. Next

1. **Get the raw StockNet price files** and establish what `data[3:6]` encode.
   This is the one gap between a result and an explanation.
2. **Run a GARCH(1,1) baseline.** Volatility forecasting has strong classical
   baselines; +0.1433 MCC means little until compared against one.
3. **Try longer horizons** (5-day, 10-day volatility), where signal-to-noise is
   conventionally better.
4. **Do not add architecture.** Eight experiments say the marginal return on
   model complexity here is zero or negative.
