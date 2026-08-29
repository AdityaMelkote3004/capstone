# Experiment 2 (Phase A) — Hierarchical Tweet-Attention Encoder

## Objective

Test the project roadmap's Experiment 4 question directly: **does hierarchical
tweet attention (tweet-level attention → daily GRU → temporal attention) over
actual tweet content outperform the flat tweet-**count** feature used in
Experiment 1, and does text-encoder choice (USE / FinBERT / VADER) matter once
aggregation stops being mean/count pooling?**

Architecture (embedding-agnostic — only the input projection's dimension
changes across backends):

```
Company tweets for (ticker, day)
        |
per-tweet embedding (USE 512-d / FinBERT CLS 768-d / VADER 4-d)
        |
   projected to a shared 64-d space
        |
Tweet-level attention  (which tweets today mattered?)
        |
   one 64-d vector per day
        |
Day-level GRU + temporal attention  (which of the last 4 days mattered?)
        |
   one 64-d text embedding per sample
```

Stage A trains this encoder + a linear head standalone on `Target`, per
backend (the direct embedding-choice ablation). Stage B freezes each trained
encoder, extracts its 64-d output, and concatenates it (late fusion, **not**
bilinear) into the same FS1–FS4 × {Logistic Regression, LSTM, MLP} harness
used in Experiment 1, on the identical official split.

## Headline result: hierarchical attention did NOT beat Experiment 1's naive tweet count

| | Best MCC |
|---|---:|
| FS3 (hierarchical attention, best of 9 backend×model combos) | **+0.0488** (VADER, Logistic Regression) |
| FS4 (hierarchical attention + fundamentals, best of 9 combos) | +0.0799 (VADER, Logistic Regression) |
| **Experiment 1's FS3 (raw tweet count, LSTM)** | **+0.0922** |

None of the 9 FS3 backend×model combinations beat Experiment 1's simple
scalar tweet-count feature. This is a genuine negative result, not a bug —
see "Why this probably happened" below for the leading hypothesis and the
follow-up experiment it motivates.

## Full results

### Stage A — standalone text-only classifier (embedding-choice ablation)

| Backend | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| USE | 0.5129 | 0.6780 | **0.0000** | 0.4559 |
| FinBERT | 0.5204 | 0.5421 | 0.0392 | 0.5183 |
| VADER | 0.4981 | 0.5031 | -0.0036 | 0.4948 |

**USE's MCC is exactly 0.0, and its AUC (0.456) is below random.** MCC is
mathematically forced to 0 when a confusion matrix has an entirely empty row
or column — i.e. this is the signature of a classifier that **collapsed to
predicting a single class for every sample**, not a classifier that learned
weak-but-real structure. Combined with F1=0.678 (consistent with always
predicting the majority-ish class) and AUC below 0.5, the most likely
explanation is that the USE-backed encoder's attention degenerated during
training rather than learning anything from tweet content. FinBERT is the
only backend with an MCC that both survives the collapse check and beats
zero, though modestly (+0.039).

### Stage B — full FS1–FS4 grid × 3 backends × 3 models (test set, official split)

| FeatureSet | Backend | Model | Accuracy | F1 | MCC | AUC |
|---|---|---|---:|---:|---:|---:|
| FS4_Price_Fundamentals_HierAttnTweets | VADER | Logistic Regression | 0.5419 | 0.6002 | **+0.0799** | 0.5405 |
| FS2_Price_Fundamentals | - | Logistic Regression | 0.5366 | 0.5947 | +0.0688 | 0.5321 |
| FS2_Price_Fundamentals | - | MLP | 0.5228 | 0.6711 | +0.0479 | 0.5246 |
| FS4_Price_Fundamentals_HierAttnTweets | FinBERT | LSTM | 0.5223 | 0.5079 | +0.0470 | 0.5272 |
| FS3_Price_HierAttnTweets | USE | Logistic Regression | 0.5234 | 0.5687 | +0.0428 | 0.5293 |
| FS4_Price_Fundamentals_HierAttnTweets | USE | LSTM | 0.5118 | 0.3637 | +0.0417 | 0.5326 |
| FS2_Price_Fundamentals | - | LSTM | 0.5237 | 0.6452 | +0.0403 | 0.5233 |
| FS4_Price_Fundamentals_HierAttnTweets | USE | Logistic Regression | 0.5180 | 0.5492 | +0.0333 | 0.5234 |
| FS3_Price_HierAttnTweets | FinBERT | LSTM | 0.5177 | 0.5442 | +0.0333 | 0.5194 |
| FS3_Price_HierAttnTweets | FinBERT | Logistic Regression | 0.5172 | 0.5594 | +0.0306 | 0.5247 |
| FS3_Price_HierAttnTweets | USE | LSTM | 0.5056 | 0.3550 | +0.0276 | 0.5317 |
| FS4_Price_Fundamentals_HierAttnTweets | FinBERT | Logistic Regression | 0.5153 | 0.5575 | +0.0268 | 0.5199 |
| FS4_Price_Fundamentals_HierAttnTweets | FinBERT | MLP | 0.5169 | 0.6683 | +0.0238 | 0.4920 |
| FS3_Price_HierAttnTweets | FinBERT | MLP | 0.5094 | 0.5201 | +0.0183 | 0.5201 |
| FS1_Price | - | MLP | 0.5145 | 0.6751 | +0.0154 | 0.4777 |
| FS4_Price_Fundamentals_HierAttnTweets | VADER | MLP | 0.5142 | 0.6413 | +0.0149 | 0.4962 |
| FS3_Price_HierAttnTweets | VADER | LSTM | 0.5091 | 0.5568 | +0.0136 | 0.5177 |
| FS4_Price_Fundamentals_HierAttnTweets | VADER | LSTM | 0.5008 | 0.5057 | +0.0018 | 0.4967 |
| FS1_Price | - | LSTM | 0.5124 | 0.6736 | -0.0004 | 0.4836 |
| FS4_Price_Fundamentals_HierAttnTweets | USE | MLP | 0.4954 | 0.4077 | -0.0009 | 0.5115 |
| FS3_Price_HierAttnTweets | VADER | MLP | 0.5038 | 0.6550 | -0.0274 | 0.5203 |
| FS3_Price_HierAttnTweets | USE | MLP | 0.4874 | 0.5572 | -0.0342 | 0.5030 |

**Best overall: FS4 [VADER] / Logistic Regression, MCC +0.0799, F1 0.6002.**
Full per-run metrics (with confusion matrices) are saved under
`results/experiment2_hierarchical_attention/{FEATURE_SET}/{model}/metrics.json`.

## Why this probably happened

The leading hypothesis is an **architecture limitation in Stage B, not a
failure of the encoder itself**: each backend's Stage-A-trained encoder was
frozen and its 64-d output was only **concatenated** to the LSTM's *final*
price hidden state, after the recurrence finished. Price dynamics and text
dynamics never interact during processing — the LSTM never sees "what the
tweets said" at any intermediate timestep, only a static summary appended at
the very end. Experiment 1's tweet count, by contrast, was fed **into the
LSTM sequence itself** as one scalar per day, letting the recurrence learn
joint price+text dynamics directly. A frozen, late-concatenated embedding is
a structurally weaker fusion than that regardless of embedding quality —
which is exactly the gap **bilinear fusion** (MAN-SF's actual mechanism,
Experiment 2's remaining piece) is designed to close.

Secondary factors likely compounding it:
- **VADER's own standalone signal is worse than noise** (Stage A MCC
  −0.0036), so its FS4 win is more likely a small-variance artifact a
  regularized Logistic Regression latched onto than real financial signal —
  treat that specific result skeptically rather than as "VADER is the best
  embedding."
- **USE's encoder appears to have collapsed** (MCC exactly 0, AUC below
  random) rather than learned anything from tweet content — see Stage A
  discussion above.
- With only ~20K training samples spread across 87 tickers and sparse tweet
  coverage per ticker-day, there may not be enough signal-bearing structure
  for the attention mechanism to learn a meaningfully better weighting than
  "how many tweets were there" — a known risk flagged before this
  experiment ran.

## What this experiment does NOT include

- **No bilinear fusion, no GAT** — text and price/fundamentals combine only
  by concatenation. This is the leading suspect for the negative result
  above and is Experiment 2's remaining scope.
- **No joint fine-tuning** — the text encoder was frozen after Stage A;
  Stage B never lets its parameters adapt jointly with price/fundamentals.
- **No event/global tweets** — only company tweets, same scope as
  Experiment 1's FS3.
- **Fundamentals are unchanged from Experiment 1** — raw 8 EDGAR features,
  zero-filled, not a learned embedding.

## Recommended next step

Before concluding "hierarchical attention doesn't help on this dataset,"
rule out that the *late-fusion-of-frozen-embeddings* design is what's
failing, not the encoder, via (in order of cost):

1. **Joint fine-tuning** — stop freezing the encoder after Stage A; let its
   parameters keep updating during Stage B training instead of using a fixed
   checkpoint.
2. **Bilinear fusion instead of concatenation** — this is Experiment 2
   proper. MAN-SF's own ablation table already shows bilinear fusion beats
   concatenation on this exact task (concatenation ≈0.156 MCC vs. bilinear
   ≈0.195 MCC), so this is independently motivated regardless of this
   experiment's outcome.

Given Experiment 2 (full MAN-SF: temporal-attention price encoder, bilinear
fusion, GAT) is the roadmap's next step regardless, folding this fix in there
is more efficient than patching Stage B's fusion mechanism twice.

## Reproducing this experiment

`preprocessing_notebooks/MMGTFFF_hierarchical_tweet_attention.ipynb` on a
Colab GPU runtime. Clones both `stocknet-dataset` and this project's
`capstone` repo (for cached EDGAR fundamentals — no new SEC API calls).
Caches each backend's raw per-tweet embeddings to
`final/tweet_embeddings_cache/{backend}_embeddings.npy` so a crash/restart
doesn't require recomputing a finished backend from scratch.
