# Experiment 3 Summary — MAN-SF Joint Architecture (Bilinear Fusion)

*Full technical write-up: `results/experiment3_mansf_bilinear/EXPERIMENT3_MANSF_BILINEAR.md`*

## What we did

Built the actual MAN-SF architecture: a GRU + temporal-attention price
encoder, the same hierarchical tweet-attention text encoder from Experiment
2, combined via **bilinear fusion**, trained **end-to-end from scratch** as
one network — no freezing, no late concatenation. Ran it across the same
three text backends (USE, FinBERT, VADER), with and without fundamentals
concatenated after fusion.

## How we did it

Structurally identical data pipeline to Experiments 1 and 2 (same exact
StockNet windowing, labeling, tweet alignment, cached fundamentals and
tweet embeddings), but a new model: `PriceEncoder` (GRU + temporal
attention) and `HierarchicalTweetEncoder` (unchanged from Experiment 2)
feed into `BilinearFusion` (`nn.Bilinear`), and the whole thing — encoders
and fusion together — is trained jointly against the direction label. This
directly tests Experiment 2's conclusion: was the previous negative result
caused by the fusion mechanism, not the text encoder?

**A bug was found and fixed along the way.** The first run fed the
classifier head *only* the bilinear fusion output. Result: Price+Text came
back with MCC exactly 0.0 for both USE and FinBERT, with **bit-for-bit
identical predictions across the two backends** — proof this was an
optimization pathology in the bilinear branch itself (three different
embeddings cannot coincidentally produce identical wrong answers), not an
embedding-quality problem. The fix: concatenate the price encoder's own
output alongside the bilinear fusion output, so the classifier head always
has a working fallback signal and can't fully collapse. Confirmed fixed —
after the fix, USE/FinBERT/VADER produce different, non-identical, non-zero
predictions.

## Why we did it this way

Fixing a training bug and confirming a hypothesis are two different
questions, and it mattered to answer both explicitly rather than declaring
victory once the collapse went away. A model that "runs without collapsing"
isn't automatically a model that "works better than baseline" — those
needed to be checked separately.

## What we found — the fusion mechanism wasn't the whole story either

The collapse bug is confirmed fixed. But **the fixed model still doesn't
beat Experiment 1's baseline** (+0.092 MCC): the best result here is plain
Price+Fundamentals with *no text at all* (+0.033 MCC), and every price+text
bilinear combination scores lower than that.

Re-reading MAN-SF's own published ablation table explained why, rather than
leaving this as an unexplained shortfall. Their reported progression shows
**GAT+price alone (no text) already reaching ≈0.117 MCC** — higher than
every price+text result either Experiment 2 or 3 produced. The
concatenation/attention/bilinear-fusion rows in their table (≈0.156 → 0.173
→ 0.195) aren't comparing fusion mechanisms on price+text alone — they're
comparing fusion mechanisms on top of the *full pipeline including GAT*.
The graph is plausibly the dominant contributor to their headline number,
not the fusion mechanism this project has spent two experiments tuning.

## How we should move forward

**Stop tuning price+text fusion in isolation and build GAT next** (sector
edges first, per the project roadmap's own sequencing). The published
evidence suggests inter-stock graph structure contributes more to
StockNet-task performance than fusion-mechanism sophistication does — two
experiments' worth of results on this project's own data are consistent
with that. Once GAT is in place on top of this now-correctly-trained
price+text branch, that's the point to re-evaluate whether bilinear fusion
(vs. simpler concatenation) actually earns its complexity — not before.
