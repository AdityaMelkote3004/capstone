# Hierarchical Tweet-Attention Encoder — Design Spec

Date: 2026-08-26

## Purpose

Replace the flat "tweet count" feature used in Experiment 1 (exact StockNet
reproduction) with an embedding-agnostic hierarchical attention encoder over
actual tweet content, and measure whether (a) attention-based aggregation and
(b) embedding choice (USE / FinBERT / VADER) improve on Experiment 1's
results. This directly answers the project roadmap's Experiment 4 question
("does hierarchical tweet attention outperform simple mean/count pooling,
and which text encoder works best once aggregation is fixed") without yet
building the full MAN-SF bilinear-fusion + GAT architecture (Experiment 2).

## Scope

In scope:
- One `HierarchicalTweetEncoder` module, parametrized by embedding backend
  (input dim varies: USE 512, FinBERT 768, VADER 4), all projected to a
  common 64-d space before attention so the attention/GRU layers are
  backend-agnostic.
- Per-tweet embedding computation for all three backends over the same
  StockNet tweet corpus used in Experiment 1.
- Stage A: standalone text-only classifier per backend (encoder + linear
  head, trained end-to-end on Target) — the embedding-choice ablation.
- Stage B: FS1–FS4 grid (Price / +Fundamentals / +Tweets / Full) × 3 models
  (Logistic Regression, LSTM, MLP) × 3 backends = 36 runs, reusing the exact
  same official split, buffer-zone filtering, and window-masking convention
  as Experiment 1. FS1/FS2 are identical to Experiment 1 (no tweet feature);
  FS3/FS4 use the frozen Stage-A-trained encoder's output as a fixed 64-d
  feature.

Out of scope (explicitly deferred to Experiment 2):
- Bilinear fusion of price and text embeddings (Stage B uses late-fusion
  concatenation only, consistent with "LSTM/MLP/LR is just for a baseline
  table" framing).
- GAT / graph structure.
- Event-conditioned attention (global/event tweets are not used at all here
  — only company tweets, matching Experiment 1's FS3 scope).
- Fundamentals encoder (FS2/FS4 fundamentals are the same raw+zero-filled
  features as Experiment 1, not a learned embedding).

## Architecture

```
Company tweets for (ticker, day)
        |
per-tweet embedding (USE 512 / FinBERT-CLS 768 / VADER 4)
        |
   Linear projection -> 64-d
        |
Tweet-level attention (learned scorer, softmax over tweets that day,
masked for days with zero tweets -> zero vector, no NaN from empty softmax)
        |
   one 64-d vector per day
        |
Day-level GRU over the (<=4) input days in the window (masked via
Window_Length, same convention as Experiment 1's price/fundamentals window)
        |
Temporal attention over GRU outputs
        |
   one 64-d text embedding per sample
```

Stage A: `HierarchicalTweetEncoder` -> `Linear(64, 2)` trained end-to-end on
Target with CrossEntropyLoss, early stopping on validation MCC (same protocol
as Experiment 1's LSTM/MLP training loop). Run once per backend.

Stage B: freeze the winning-per-backend encoder (or all three, since the
user wants all three carried through the full grid), extract each sample's
64-d embedding, and use it as an additional feature in the same
FS1–FS4/LR-LSTM-MLP harness from Experiment 1:
- LR / MLP: flatten the window (price [+fundamentals]) and concatenate the
  64-d tweet embedding at the end (same as Experiment 1's flat-feature
  convention, just replacing 1 scalar with 64 numbers).
- LSTM: run price [+fundamentals] through the masked LSTM to get its final
  hidden state, concatenate the 64-d tweet embedding to that hidden state,
  then classify. (Late-fusion by concatenation — NOT bilinear fusion, which
  is explicitly Experiment 2's job.)

## Data flow / sample construction

Reuses `build_samples_for_ticker`'s windowing, buffer-zone filter, and
tweet-to-trading-day alignment logic from the Experiment 1 notebook
unchanged, but additionally retains the raw token lists for each input
day's company tweets (capped at StockNet's own `max_n_msgs=30` per day),
instead of collapsing them to a scalar count.

## Error handling

- Days with zero tweets: tweet-level attention returns a zero vector
  (mask-guarded, not a softmax over an empty/all-masked set).
- Tickers/backends with fetch failures (e.g. TF-Hub download hiccup):
  notebook should fail loudly rather than silently degrade to zero
  embeddings, since a silent all-zero backend would look like "this
  embedding doesn't help" when it's actually "this embedding never ran."

## Reproducibility / environment

Colab GPU runtime. Clones both `stocknet-dataset` (raw prices/tweets) and
this project's own `capstone` GitHub repo (for `dataset/final/` cached EDGAR
fundamentals, reused unchanged from the corrected preprocessing notebook —
no new SEC API calls needed). PyTorch for the encoder/training loop,
TensorFlow Hub for USE, `transformers` for FinBERT, `vaderSentiment` for
VADER (all standard Colab-installable, no local-machine dependency).

## Validation plan

Given USE/FinBERT require large model downloads not practical to run
against the full ~100K-tweet corpus on the local dev machine, local
validation covers: (a) the `HierarchicalTweetEncoder` module's forward pass
and masking behavior against synthetic embeddings of each backend's
dimensionality, (b) the VADER backend end-to-end (cheap, no download,
validates the full pipeline including sample construction, encoder
training, and the FS1–FS4 grid harness), (c) syntax/compile checking of
every notebook cell. Full USE/FinBERT runs are validated by the user in
Colab, per the pattern established for Experiment 1.
