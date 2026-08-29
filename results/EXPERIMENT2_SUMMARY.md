# Experiment 2 Summary — Hierarchical Tweet-Attention Encoder (Phase A)

*Full technical write-up: `results/experiment2_hierarchical_attention/EXPERIMENT2_HIERARCHICAL_ATTENTION.md`*

## What we did

Replaced Experiment 1's flat tweet **count** feature with an actual
**hierarchical attention encoder** over tweet content — tweet-level
attention (which tweets today mattered) feeding a day-level GRU + temporal
attention (which of the last 4 days mattered) — and tested it with three
different text-embedding backends: USE (512-d, what MAN-SF's paper uses),
FinBERT (768-d, financial-domain-tuned), and VADER (4-d lexicon sentiment,
a cheap control/lower bound).

## How we did it

The encoder was built to be **embedding-agnostic**: the same
tweet-attention + GRU + temporal-attention architecture runs on any backend,
only the input projection's dimension changes. This makes the 3-way backend
comparison a controlled ablation rather than three different builds.

Training happened in two stages:
1. **Stage A**: each backend's encoder + a linear head trained standalone on
   the direction label — the direct embedding-choice ablation.
2. **Stage B**: each trained encoder was **frozen**, its 64-d output
   extracted, and **concatenated** (late fusion) onto the same FS1–FS4 ×
   {Logistic Regression, LSTM, MLP} grid used in Experiment 1, so results
   were directly comparable.

## Why we did it this way

Experiment 1's tweet-count feature is a strong signal (best overall result)
but a crude one — it can't distinguish "50 tweets about an earnings beat"
from "50 tweets that are noise." Your own project roadmap's prior FinBERT
experiments (mean-pooling era) found that swapping embedding dimensionality
did nothing — but that was under mean-pooling, which structurally destroys
whatever an embedding captures by averaging everything together. Attention
is *capable* of selecting signal instead of averaging it away; this
experiment tested whether that capability actually pays off once given the
chance, and which embedding backend benefits most from it.

## What we found — a real negative result

**None of the 9 backend×model combinations for hierarchical attention beat
Experiment 1's naive tweet count** (best here: +0.049 MCC, VADER/Logistic
Regression, vs. Experiment 1's +0.092). Two specific findings explain why,
rather than leaving it a mystery:

- **USE's standalone encoder collapsed** — Stage A MCC was exactly 0.0 and
  AUC below random, the signature of a classifier that settled on predicting
  one class for every sample rather than learning real structure.
- **The Stage B fusion mechanism (frozen + late concatenation) never let
  text and price interact during training.** The LSTM saw price sequentially
  but only received the text embedding as a single number bolted onto its
  final hidden state — a structurally weak fusion regardless of how good the
  embedding was.

## How we should move forward

This result pointed at the fusion mechanism, not the encoder, as the likely
culprit — which became the hypothesis Experiment 3 was built to test: keep
the same hierarchical-attention text encoder, but train it **jointly**
end-to-end with the price encoder via **bilinear fusion** instead of
freezing it and concatenating. See `EXPERIMENT3_SUMMARY.md` for whether that
hypothesis held up.
