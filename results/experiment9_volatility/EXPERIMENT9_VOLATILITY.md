# Experiment 9 — Volatility is predictable; our model is not the reason

**Status:** Complete. 20 cells (4 rows × 5 seeds) on the volatility target,
compared against 65 direction-target cells from Experiment 8.
**Run:** `scripts/run_experiment8_lowdim_text.py --target volatility`
**Raw results:** `results/experiment8_lowdim_text/results_volatility.csv`
**Mechanism checks:** `MECHANISM_NOTES.md` (read this before citing the result)
**Classical baselines:** `garch_baseline.json`, `scripts/run_garch_baseline.py`

---

> ## The headline, stated correctly
>
> Volatility **is** predictable on this data where direction is not — that part
> holds, and two independent method families agree on it.
>
> But the neural model is **not** the reason. EWMA (RiskMetrics, 1994) — one
> parameter, λ=0.94, fed to a logistic regression as a single feature — beats
> our LSTM on **every metric**, and on within-day MCC by a wide margin
> (**+0.1966 vs +0.1322**). See §2.2.
>
> And adding text to that baseline **actively harms it**: +0.1475 → +0.0925.

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

### 2.1 Text still adds nothing to OUR model

| vs price-only, volatility target | Pooled | p | Within-day | p |
|---|---|---|---|---|
| + tweet counts | −0.0328 | 0.295 | −0.0199 | 0.421 |
| + counts + PCA-2 | +0.0060 | 0.793 | +0.0150 | 0.454 |
| + counts + PCA-8 | +0.0012 | 0.958 | +0.0113 | 0.601 |

Adding FinBERT to a volatility model changes nothing (p = 0.96). This is the
same answer Experiments 1, 2, 3, 7 and 8 gave for direction, now confirmed on a
target where the model demonstrably *can* learn.

### 2.2 The classical baselines beat us

Volatility forecasting has thirty years of standard methods. Measured on the
same test set, same target, same threshold:

| Baseline | Pooled MCC | Within-day | Acc | AUC |
|---|---|---|---|---|
| realised vol, 5d | +0.0837 | +0.1267 | 55.4% | 0.569 |
| realised vol, 10d | +0.1218 | +0.1712 | 57.7% | 0.588 |
| realised vol, 20d | +0.1211 | +0.1810 | 58.4% | 0.602 |
| **EWMA (RiskMetrics, λ=0.94)** | **+0.1475** | **+0.1966** | **59.6%** | **0.608** |
| GARCH(1,1) | +0.0905 | +0.1710 | 44.7% | 0.592 |
| all classical combined | +0.1424 | +0.1686 | 57.7% | 0.595 |
| *Exp 9 LSTM (price only)* | *+0.1433* | *+0.1322* | *58.5%* | *0.600* |

**EWMA beats the LSTM on all four metrics**, using one parameter and no
training. Three of five classical baselines beat it on within-day. The LSTM is
not competitive with a one-line formula.

GARCH(1,1) underperforming EWMA is unremarkable on ~256 training observations
per ticker; its low accuracy against a respectable AUC (0.5915) is a
threshold-calibration artefact, not absent signal.

### 2.3 Text hurts the best baseline

The sharpest test of the text question: bolt text onto EWMA rather than onto our
own models.

| Model | Pooled MCC | Within-day |
|---|---|---|
| EWMA alone | **+0.1475** | **+0.1966** |
| EWMA + tweet counts | +0.1415 | +0.1893 |
| EWMA + PCA-8 text | **+0.0925** | +0.1148 |
| EWMA + counts + text | +0.0987 | +0.1423 |

Eight dimensions of FinBERT cost a one-parameter model **37% of its MCC** and a
third of its within-day score. Across this project text has been uninformative;
here it is actively harmful — consistent with the sample-size argument in
`src/models/lowdim_text.py`, since every added dimension is another parameter
estimated from the same 20,344 training rows.

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

**Benchmark claims.** None against the literature. Published StockNet/MAN-SF
figures are *direction* results and are not comparable to a volatility target.
The relevant baselines are the classical volatility forecasts in §2.2, and the
LSTM loses to them.

**The seed spread is large** (0.065–0.100 pooled). Every row is strongly
positive on all five seeds, but individual cell values should not be quoted to
four decimals.

## 5. What this means for the project

Three findings, which fit together into one claim:

1. **Next-day direction is not predictable** from this data. Seven experiments,
   five architectures — and in each case where a positive result appeared, the
   mechanism was identified: seed luck (Exp 1), feature/label timing overlap
   (Exp 7), missingness encoding (Exp 8).
2. **Next-day volatility is predictable** — but by EWMA (+0.1475), not by us
   (+0.1433 pooled, +0.1322 within-day). No neural model built in this project
   beat a classical baseline on either target.
3. **Financial text adds nothing, and subtracts from a good model.** Every
   apparent text gain across eight experiments traced back to volume, coverage,
   or timing — never semantic content — and against the best baseline text costs
   37% of its MCC.

The honest headline is not "our model beat the benchmark" and not "nothing
worked." It is: **the ceiling was set by the question, and the answer to the
answerable question was already available in 1994.** That is a real, controlled
finding about the marginal value of deep learning and of financial text on this
benchmark — and it is only credible because the classical baseline was run.

## 6. Next

1. **Get the raw StockNet price files** and establish what `data[3:6]` encode.
   This is the one gap between a result and an explanation.
2. ~~Run a GARCH(1,1) baseline.~~ **Done — §2.2. It changed the conclusion.**
3. **Try longer horizons** (5-day, 10-day volatility), where signal-to-noise is
   conventionally better — and always against EWMA, never against price-only.
4. **Do not add architecture.** Nine experiments say the marginal return on model
   complexity here is zero or negative. The open question worth effort is whether
   *any* learned model can beat EWMA on this data, and the honest prior is no.
