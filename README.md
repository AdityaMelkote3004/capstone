# MMGTFFF — Multi-Modal Graph Transformer for Federated Financial Forecasting

PES University capstone PW26\_MHR\_03 (guide: Dr. Mamatha H R).
Team: Aditya M Hubli, Achintya N, Aditya Melkote, Aayush Prem.

Next-day stock movement prediction (up / down) on the **StockNet** benchmark:
87 US stocks, 2014–2015, test period Oct–Dec 2015 (3,720 predictions).

## The result in one paragraph

A transformer in which every stock attends to every other stock each day
(the DTML backbone, KDD 2021), extended with a multi-modal profile per stock
(volatility, SEC EDGAR fundamentals, tweet volume, and peer features from a
stock graph), reaches a pooled MCC well above every baseline we trained on the
same split. Pooled MCC, however, mixes two skills: calling the whole market's
direction on a day, and ranking stocks against each other. The backbone's
shared market signal makes it a market-timer with no stock-selection skill;
MMGT keeps the timing gain, and its multi-modal profile restores stock selection
to the level of a per-stock LSTM. Hand-built graphs inside the attention layer add
nothing. Federated training across sector clients scores lower under a fixed
schedule, mostly because it stopped near its first round (a one-epoch centralised
control looks the same).

**Paper:** [`paper/main.tex`](paper/main.tex) (IEEE conference format).
**All numbers:** [`results/experiment11_dtml/RESULTS.md`](results/experiment11_dtml/RESULTS.md).
**Protocol and its history:** [`results/experiment11_dtml/PREREGISTRATION.md`](results/experiment11_dtml/PREREGISTRATION.md).
**How the project got here (Experiments 1–10):** [`results/CAPSTONE_FINDINGS.md`](results/CAPSTONE_FINDINGS.md).

## Repository map

| Path | What |
|---|---|
| `paper/` | `main.tex`, `refs.bib`, figures, `make_figures.py`, `numbers.json`, `REFERENCE_CHECK.md` |
| `src/data/acl18.py` | StockNet in the Adv-ALSTM preprocessing (the data DTML reports on) |
| `src/data/acl18_extra.py` | the multi-modal stock profile (leak-free v2) |
| `src/data/acl18_graph.py` | daily stock graph: sector, correlation, co-mention, market-residual |
| `src/models/dtml.py` | DTML backbone + MMGT extensions (profile, graph-biased / masked attention) |
| `scripts/run_dtml*.py` | training and evaluation (test locked unless `--eval-test`) |
| `scripts/run_acl18_baselines.py` | logistic regression and MLP baselines |
| `scripts/runs/` | the exact shell commands for every stage of Experiment 11 |
| `tests/test_profile_leakage.py` | proves the profile uses only data dated before the target day |
| `results/experiment1_*` … `experiment10_*` | earlier experiments, kept as the original record |

## Reproducing the paper

```bash
pip install -r requirements.txt   # plus torch, torch_geometric
# 1. data: clone github.com/fulifeng/Adv-ALSTM and copy
#    data/stocknet-dataset/price/{ourpped,trading_dates.csv} to data/acl18/
python tests/test_profile_leakage.py          # must print PASS
bash scripts/runs/run_dtml_10seed.sh           # DTML backbone (seeds 5-9; 0-4 in run_dtml_final.sh)
bash scripts/runs/run_acl18_baselines.sh       # baselines
bash scripts/runs/run_clean_rerun.sh           # MMGT, ablations, federated
python paper/make_figures.py                   # figures
```

Everything runs on CPU; one model takes 5–15 minutes.

## Notes on honesty

Five earlier results in this project looked significant and were later shown
to be artefacts (see `results/CAPSTONE_FINDINGS.md`, section 2). A sixth, a
target-day leak in the first version of the stock profile, was caught before
publication; the paper reports only the corrected results.
