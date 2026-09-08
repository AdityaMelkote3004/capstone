# Experiment 6 — Dense Stock Graph + Model Grid

**Status:** Complete. 84/84 grid cells trained (4 models × 7 edge combinations × 3 seeds).
**Run:** local CPU (PyTorch 2.11.0+cpu, no GPU), `scripts/run_dense_graph_grid.py --pass 3`.
**Raw results:** `results.csv` (84 rows) + one `metrics.json` per run.

---

## 1. Objective

Experiments 4 (GAT) and 5 (graph transformer) both ended on the same open
question: they used a **sector-only** graph — 9 disjoint complete cliques —
and could not say whether their near-zero results reflected a weak *model*
or a weak *graph*. Experiment 6 answers the graph half of that question by
holding the data, splits, and training protocol fixed while varying the
graph itself.

Three edge types were built, every non-empty combination of them was run
(7 combos), across four graph models, at three seeds each.

## 2. The MAN-SF non-reproduction caveat

**None of the graphs built here is MAN-SF's graph.** MAN-SF's reported
GAT+price MCC of 0.117 was measured on a **Wikidata company-relation
graph** — curated corporate relationships (shared officers, ownership,
industry ontology links). This experiment builds sector cliques, rolling
return correlation, and tweet co-mention. No result in this document may
be described as reproducing, approaching, or beating MAN-SF's 0.117; the
graphs are not comparable, so the numbers are not either.

## 3. The three edge types, as measured

Measured over all 504 trading days from the built edge cache. These
supersede the pre-measurement estimates in the design spec.

| Relation | Mean edges/day | Range | Days with ≥1 edge |
|---|---|---|---|
| `sector` | 381.0 | 381–381 | 504 / 504 |
| `correlation` (60d window, top-5) | 321.0 | 0–356 | 494 / 504 |
| `comention` (5-day tweet lookback) | 10.8 | 0–71 | 465 / 504 |

- **Sector** is static: 9 complete cliques, 381 edges, identical every day.
  This is exactly Experiment 4's graph, so sector-only cells re-run that
  experiment under this protocol.
- **Correlation** is dense and dynamic, and needs a warm-up: the first
  **10** trading days have insufficient history for a 60-day window and
  correctly degrade to zero edges rather than crashing.
- **Co-mention is very sparse** — 10.8 edges/day on average against 381
  sector edges. This matters for reading the results: a relation
  contributing ~3% of the edge mass cannot be expected to move metrics
  much, and it does not.

Structure figures: `figures/graph_sector.png`,
`figures/graph_correlation.png`, `figures/graph_comention.png`,
`figures/edge_density_over_time.png`.

## 4. Results

Test MCC, mean ± std across 3 seeds. All 84 runs scored **n_samples =
3720**, matching Experiments 1/3/4/5.

### Reference points

| Reference | MCC | Accuracy | F1 |
|---|---|---|---|
| Experiment 1 (LSTM + tweets) | **+0.0922** | — | — |
| Experiment 3 (price-only) | +0.0108 | — | — |
| Experiment 4 (sector GAT, single seed) | +0.0278 | — | — |
| Constant always-UP | 0.0 | 0.5129 | 0.6780 |
| Constant always-DOWN | 0.0 | 0.4871 | 0.0 |

### Top 8 cells

| Model | Edges | MCC (mean ± std) | AUC | Acc | F1 |
|---|---|---|---|---|---|
| RGCN | sector+correlation | +0.0677 ± 0.0255 | 0.5512 | 0.5315 | 0.5208 |
| RGCN | sector+correlation+comention | +0.0497 ± 0.0066 | 0.5356 | 0.5229 | 0.5101 |
| GraphSAGE | correlation+comention | +0.0438 ± 0.0019 | 0.5302 | 0.5241 | 0.6549 |
| GraphSAGE | correlation | +0.0433 ± 0.0100 | 0.5303 | 0.5245 | 0.6526 |
| RGCN | correlation | +0.0392 ± 0.0080 | 0.5384 | 0.5219 | 0.6624 |
| RGCN | correlation+comention | +0.0326 ± 0.0103 | 0.5405 | 0.5199 | 0.6632 |
| RGCN | sector+comention | +0.0275 ± 0.0172 | 0.5326 | 0.5179 | 0.6655 |
| GraphSAGE | sector+comention | +0.0263 ± 0.0087 | 0.5142 | 0.5170 | 0.6691 |

### Aggregates

| Model | MCC (mean ± std over 21 runs) | | Edge combo | MCC (mean ± std over 12 runs) |
|---|---|---|---|---|
| RGCN | +0.0326 ± 0.0250 | | sector+correlation | +0.0319 ± 0.0275 |
| GraphSAGE | +0.0251 ± 0.0162 | | sector+corr+comention | +0.0253 ± 0.0198 |
| GAT | +0.0160 ± 0.0117 | | correlation | +0.0255 ± 0.0182 |
| GraphTransformer | +0.0133 ± 0.0183 | | correlation+comention | +0.0251 ± 0.0164 |
| | | | sector+comention | +0.0220 ± 0.0149 |
| | | | sector | +0.0164 ± 0.0140 |
| | | | comention | +0.0060 ± 0.0173 |

Result figures: `figures/mcc_by_model_and_edges.png`,
`figures/seed_variance.png`.

## 5. Honest reading — did the richer graph help?

**No distinguishable improvement.** The evidence:

**Seed spread swamps the differences.** Mean within-cell seed spread
(max−min MCC) is **0.0247**, median 0.0193, max 0.0579. The gap between
the #1 and #3 cells is 0.024 — *smaller than the average noise floor*.
The leading cell, RGCN [sector+correlation], has seeds of +0.0437,
+0.0945, +0.0650: a 0.0508 spread on a 0.0677 mean. That cell is not a
finding; it is one lucky seed and two ordinary ones.

**AUC says most cells learned nothing.** MCC is threshold-dependent and
flatters cells that happen to sit near a good cut point; AUC is not. Of
28 (model, edge) cells, **8 score AUC < 0.51** — indistinguishable from a
coin flip — and only 12 exceed 0.53. GAT [sector] averages **AUC 0.4913,
below chance**, so its +0.0160 MCC is threshold luck rather than signal.
Most cells also sit at F1 0.66–0.68 with accuracy ~0.515, which is the
**always-UP constant predictor** (0.5129 / 0.6780) wearing a graph.

**The top cells are at a different operating point, not strictly better.**
RGCN [sector+correlation] reaches its MCC by trading F1 down hard — 0.5208
mean, and 0.4043 on one seed — against the always-UP 0.6780. It predicts
DOWN more often, which helps MCC on a near-balanced target without
demonstrating better ranking (AUC 0.5512).

**What does hold up, weakly:** correlation edges are consistently better
than co-mention edges. Every combination containing `correlation` averages
+0.025 to +0.032, while `comention` alone averages **+0.0060** and is the
worst combo overall. Given co-mention supplies only ~3% of edge mass, this
reads as "the sparse tweet relation contributed nothing measurable,"
which is a real if unexciting result. Likewise RGCN (+0.0326) and
GraphSAGE (+0.0251) rank above GAT (+0.0160) and GraphTransformer
(+0.0133) — but those model gaps are again within one seed spread.

**Experiment 4's headline does not survive multi-seed re-running.** Under
this experiment's training protocol, GAT+sector scores **+0.0160 ± 0.0161**
across three seeds (+0.0263 / −0.0026 / +0.0243). Experiment 4's published
single-seed **+0.0278** sits inside that range — it was near the top of its
own seed distribution. It should not be carried forward as this project's
best graph result. (This is a statement about seed variance, not an audit
of Experiment 4's configuration; the two were not verified to be
hyperparameter-identical.)

**Nothing here beats Experiment 1.** The LSTM+tweets baseline at +0.0922
remains the project's best result. Exactly one of 84 runs (a single seed of
RGCN [sector+correlation], +0.0945) exceeded it, which at this noise level
is what you expect from 84 draws.

## 6. Leakage audit

The spec's tripwire — any cell exceeding MCC 0.15 is treated as suspected
leakage to be re-audited, not a success — **was checked and did not fire**.
The maximum MCC across all 84 runs is **+0.0945**.

Leakage is additionally a tested property, not an assumption. Every edge
builder takes an explicit cutoff and has a test asserting that corrupting
the target day and everything after it cannot change that day's edges
(`tests/test_graph_edges.py::test_leakage_target_day_cannot_change_edges`,
`::test_leakage_target_day_tweets_are_excluded`). The full suite — 57 tests
— passes.

## 7. Disclosure — uncontrolled differences and weaknesses

1. **Dev split is small.** 398 train / **42 dev** / 64 test daily
   snapshots. Model selection is early stopping on dev MCC computed from
   only 42 days, so the selection criterion is itself noisy. This is a
   plausible primary driver of the large seed spreads reported above.
2. **14 of 84 runs early-stopped with `best_epoch < 3`** — the restored
   checkpoint was chosen before the model meaningfully trained. Mean
   `best_epoch` was 13.5 (max 82). Those cells' numbers describe a barely
   trained model.
3. **Epoch budget is not controlled across models.** Every cell used
   max 100 epochs with patience 10, but the models converge at different
   rates, so they did not receive equal effective training.
4. **17 of 26,619 exact-protocol samples** are not covered by the join to
   the corrected modeling set. They degrade to empty neighbourhoods rather
   than being dropped, so they are scored but graph-less.
5. **Correlation warm-up:** the first 10 trading days have no correlation
   edges at all, so early training days see a sparser graph than the rest.
6. **R-GCN always allocates 3 relations** even when the combo uses fewer;
   unused relation weights receive no gradient. Harmless, but it means
   parameter count is constant across combos rather than matched to them.
7. **GraphTransformer [sector] and [sector+comention] score identically**
   on MCC/F1 (+0.0145 / 0.3514), differing only in AUC (0.5217 vs 0.5226).
   The per-relation attention bias initializes at zero, so adding a very
   sparse relation is close to a no-op at initialization — consistent with
   co-mention's measured 10.8 edges/day, but worth noting as a near-
   degenerate pair rather than two independent measurements.

## 8. How we should move forward

**Operational Pass-4 target: GraphSAGE [correlation].** Applying the
spec's documented tie-break literally: the leading cell is RGCN
[sector+correlation] at +0.0677 ± 0.0255, giving a tie band of ≥ 0.0422,
which contains four cells — RGCN [sector+correlation] (2 edge types), RGCN
[sector+correlation+comention] (3), GraphSAGE [correlation+comention] (2),
and **GraphSAGE [correlation] at +0.0433 ± 0.0100 (1 edge type)**. Fewer
edge types wins the tie, so the simplest cell in the band is the pick.
It is also the more defensible one: its seed spread is 0.0100 against the
RGCN leader's 0.0255.

**This is a nomination, not a claim of improvement.** Section 5 stands:
no cell in this grid is distinguishably better than the sector-only
baseline, and the tie-break exists to choose a Pass-4 starting point, not
to declare a winner.

Recommended next steps, in priority order:

1. **Fix the evaluation before adding capacity.** The 42-day dev split is
   the weakest link in this experiment — it makes every number noisier
   than the effects being measured. Consider walk-forward validation or a
   larger dev window before running Pass 4; otherwise Pass 4 inherits the
   same noise floor and will be equally unable to resolve a real effect.
2. **Run Pass 4 (text encoders) on GraphSAGE [correlation]**, per the
   deferred plan. Experiment 1 (+0.0922) says the signal in this dataset
   is in the *text*, not the graph — every graph variant tried here lands
   at half that or less. Pass 4 is the one direction with prior evidence
   behind it.
3. **Report seed spread alongside every future number.** This experiment's
   main methodological lesson is that single-seed graph results in this
   project are not trustworthy: Experiment 4's +0.0278 looked like a
   result and was within noise of +0.0160.
4. **Drop co-mention edges** unless the tweet corpus is expanded. At 10.8
   edges/day they cannot carry signal, and they cost a relation slot.
