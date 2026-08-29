# Experiment 3 — MAN-SF Joint Architecture (GRU + Attention Price, Hierarchical Attention Text, Bilinear Fusion)

## Objective

Fix Experiment 2 Phase A's negative result (frozen text encoder + late
concatenation lost to a naive tweet count) by building the actual MAN-SF
architecture: a GRU + temporal-attention price encoder and the same
hierarchical tweet-attention text encoder, combined via **bilinear fusion**,
trained **end-to-end from scratch** — no freezing, no late concatenation.

```
        PRICE                              TWEETS
          |                                   |
         GRU                          per-tweet embedding
          |                       (USE / FinBERT / VADER)
   Temporal Attention                         |
          |                          Tweet-level Attention
          |                                   |
          |                            Daily Tweet Vector
          |                                   |
          |                                 Day GRU
          |                                   |
          |                       Temporal Attention (text)
          |                                   |
          +----------------+------------------+
                            |
                    BILINEAR FUSION
                            |
                    Stock Representation
                            |
                      Classifier Head
                            |
                        UP / DOWN
```

Scope: no GAT/graph yet (this is MAN-SF's per-stock branch only), no
event/global tweets, fundamentals are a project-specific extension
concatenated after fusion (not part of real MAN-SF).

## Bug found and fixed: classifier-head collapse in the bilinear branch

The first run (`pre_fix_buggy_run/`) fed the classifier head **only**
`self.fusion(price_repr, text_repr)` when text was included. Result:
**Price+Text (bilinear) came back with MCC exactly 0.0 for both USE and
FinBERT, with bit-for-bit identical accuracy and F1 to 16 decimal places
across the two backends** (VADER landed on the same pattern, off by 1-2
samples). Three different text embeddings producing bit-identical wrong
predictions is proof this was an optimization pathology in the bilinear
branch itself, not an embedding-quality problem — the model learned to
ignore the text branch entirely and output a constant class.

Confirming evidence: the collapse disappeared the moment fundamentals were
concatenated after fusion (MCC +0.041 USE, +0.060 FinBERT, +0.041 VADER in
the buggy run) — the head had *some* real per-sample signal to learn from
even when the bilinear branch was degenerate. Price-only and
Price+Fundamentals never touch `BilinearFusion` and were never at risk,
consistent with them being the only two non-degenerate results in that run.

**Fix**: concatenate `price_repr` — already shown to train to a real,
non-degenerate MCC on its own — alongside the bilinear fusion output,
instead of routing everything through the bilinear branch exclusively. This
gives the classifier head a working fallback signal so it can't fully
collapse even if the bilinear branch struggles early in training.

## Results after the fix (`final_run/`)

| Model | Backend | Accuracy | F1 | MCC | AUC |
|---|---|---:|---:|---:|---:|
| **Price + Fundamentals** | - | 0.5175 | 0.5380 | **+0.0334** | 0.5205 |
| Price + Text (bilinear) + Fundamentals | USE | 0.5194 | 0.6684 | +0.0332 | 0.5087 |
| Price + Text (bilinear) + Fundamentals | FinBERT | 0.5175 | 0.5931 | +0.0277 | 0.5135 |
| Price + Text (bilinear) | FinBERT | 0.5140 | 0.6781 | +0.0204 | 0.4986 |
| Price + Text (bilinear) + Fundamentals | VADER | 0.5161 | 0.6630 | +0.0196 | 0.5237 |
| Price + Text (bilinear) | USE | 0.5134 | 0.6781 | +0.0144 | 0.5181 |
| Price only | - | 0.5140 | 0.6720 | +0.0108 | 0.5425 |
| Price + Text (bilinear) | VADER | 0.4976 | 0.3961 | +0.0048 | 0.5254 |

**The collapse is confirmed fixed**: USE, FinBERT, and VADER now produce
different, non-identical, non-zero hard predictions (MCC +0.0144 / +0.0204 /
+0.0048 respectively) instead of the previous bit-identical MCC=0 pattern.

**But nothing here beats baseline.** Every price+text combination is well
below Experiment 1's naive tweet-count baseline (LSTM, MCC +0.092), and the
best model overall is plain Price+Fundamentals (+0.033) — no text signal at
all. Fixing the collapse replaced "collapsed to a constant" with
"trains to weak-but-real signal that still loses to a simpler baseline."

## Why: re-reading MAN-SF's own ablation table

MAN-SF's published ablation progression:

| Component | MCC |
|---|---:|
| LSTM + price only | ~0.002 |
| GRU + social text | ~0.077 |
| GCN + price | ~0.093 |
| GRU + USE text | ~0.101 |
| **GAT + price** | **~0.117** |
| Concatenation (full pipeline incl. GAT) | ~0.156 |
| Attention fusion (full pipeline incl. GAT) | ~0.173 |
| Bilinear fusion (full pipeline incl. GAT) | ~0.195 |

**GAT + price alone — no text at all — already beats every price+text
combination built in this experiment.** The concatenation/attention/bilinear
fusion rows in the published table are not comparing fusion mechanisms on
price+text alone; they're comparing fusion mechanisms *on top of the full
pipeline including GAT*. This means the graph is plausibly the dominant
contributor to MAN-SF's headline ~0.195 number, not the fusion mechanism
this experiment has been tuning. Two components (price, text) that
contribute the least in the published ablation are exactly the two this
project has built so far, without the graph.

## Recommendation: prioritize GAT next, not further fusion tuning

Continuing to tune the price+text fusion mechanism in isolation is unlikely
to close the gap to baseline, let alone to MAN-SF's full number, based on
the published ablation's own evidence. The next step should be adding GAT
(sector edges first, per the project roadmap's sequencing) on top of this
now-correctly-trained price+text branch, rather than further iterating on
the bilinear fusion mechanism itself.

## Files

- `final_run/` — post-fix results, per-model `metrics.json` (matching the
  convention used in Experiments 1 and 2), plus `full_results.json` and
  `summary.csv`.
- `pre_fix_buggy_run/` — the original collapsed run, kept for the record as
  the evidence behind the bug diagnosis above. **Do not cite these numbers
  as a real result** — they document a training pathology, not a model
  comparison.

## Reproducing this experiment

`preprocessing_notebooks/MMGTFFF_mansf_joint_bilinear.ipynb` on a Colab GPU
runtime. Same data pipeline as Experiments 1 and 2 (exact StockNet
windowing/labeling/tweet-alignment, cached EDGAR fundamentals, cached
per-tweet embeddings).
