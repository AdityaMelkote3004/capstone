# Experiment 10 — handoff

**Read this before citing any number from this folder.**

## What was found

On the **volatility** target (is tomorrow's |move| above the training median?),
graph-derived **peer features** improve within-day MCC by **+0.0287
(p = 0.0052, 5 seeds)**, while **message passing contributes nothing**.

GAT, within-day MCC, 5 seeds per cell:

| Variant | No-graph | Graph | graph vs no-graph |
|---|---|---|---|
| plain | +0.1828 | +0.1821 | −0.0007 (p=0.890) |
| cross-sectional norm | +0.1857 | +0.1840 | −0.0017 (p=0.764) |
| contrast / high-pass | — | +0.1786 | −0.0035 vs plain (p=0.517) |
| **+ peer features** | **+0.2115** | **+0.2261** | +0.0146 (p=0.069) |

Best cell: within-day **+0.2261**, pooled **+0.2805**, accuracy **65.4%**.
Reference — EWMA (RiskMetrics) baseline: within-day +0.1966, pooled +0.1475,
accuracy 59.6%.

Interpretation: a GNN averages each node toward its neighbours, which destroys
the differences a within-day ranking metric scores. The same neighbourhood
information delivered as explicit features works. The edges hold signal; message
passing was the wrong reader.

## THE BLOCKING CAVEAT

**The peer feature block bundles a sector one-hot with the peer aggregates, and
no ablation separates them.** If most of the +0.0287 is sector membership, the
finding is "sector identity helps" — not "graph neighbourhoods help" — which is
a substantially weaker claim.

Run this before citing anything:

```bash
# peer aggregates WITHOUT sector one-hot, vs sector one-hot alone
python scripts/run_experiment10_multimodal.py --full gat --targets volatility \
    --variants plain --peer --seeds 0 1 2 3 4
```
(the split requires editing `build_peer_features` in `src/data/node_features.py`
to emit the two blocks separately — they are currently concatenated)

## Also unresolved

1. The effect (+0.0287) is **below the peer row's own seed spread (0.0352)**
   despite p = 0.0052. Small n.
2. **Direction shows no peer benefit** (graph vs no-graph −0.0014, p = 0.871),
   but its matched 5-seed plain baseline was never run. `run_exp10_gaps.sh`
   does that.
3. `contrast` no-graph cells are absent by design — contrast lives inside the
   conv loop, so it is a no-op with the graph off.

## Why the bar here is high

This project produced **five** results that passed significance testing, seed
averaging, Bonferroni correction and a leakage tripwire, and were still false —
each caught later by a cheap control that was not in the original protocol. See
`results/CAPSTONE_FINDINGS.md` §2. Experiment 10's peer result is the most
promising thing in the project *and* it has an unrun control. Treat accordingly.

## Files

| Path | What |
|---|---|
| `results.csv` | all 59 cells |
| `scripts/analyze_experiment10.py` | regenerates every number above |
| `src/data/node_features.py` | 39 base features + 16 peer/sector features |
| `src/models/multimodal_graph.py` | 4 convs, `use_graph` control, contrast, cross-sectional |
| `run_exp10_full.sh` | the four-stage run |
| `run_exp10_gaps.sh` | the missing direction baseline |

Leakage: max \|corr\| between any feature and the target day's move is **0.032**.
`n_samples` is **3720** on every cell, matching Experiments 1 and 4–9.
