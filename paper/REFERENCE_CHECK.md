# Reference verification log (2026-11-10)

All 45 entries in `refs.bib` are cited in `main.tex`, and every citation resolves.
Each entry was checked against an authoritative record. "Verified" means title,
authors, venue, volume/issue and pages were matched to that source.

## Verified against Crossref (api.crossref.org/works/<DOI>)
xu2018stocknet, feng2019advalstm, yoo2021dtml, sawhney2020mansf, sawhney2021fast,
li2024master, feng2019rsr, chen2018gcn, cheng2022multimodality,
zhang2025gnnsentiment, ang2022guided, zhang2025camef, zhu2025msgformer,
nguyen2025dasf, li2024finreport, liu2025multimodal, kumarappan2024federated,
xia2025fingraphfl, zhong2025financial, aljunaid2025secure, hochreiter1997lstm,
burges2005ranknet, matthews1975, chicco2020mcc, welch1947, kunsch1989, nosek2018

## Verified against the arXiv API (export.arxiv.org): title, authors, ID
kumar2024dynamic, sonani2025hybrid, hu2025differential, harit2024breaking,
wang2021review, qayyum2025news, dukkipati2025predictive, niu2025ngat,
noseda2025federated, tan2025graphprivacy

## Verified against the publisher page
- mcmahan2017fedavg: proceedings.mlr.press/v54 (PMLR 54, pp. 1273-1282, 2017)
- kingma2015adam: arXiv 1412.6980 comment field ("Published as a conference paper at the 3rd ICLR, San Diego, 2015")

## Partially verified — please confirm by hand
| Key | Verified | Not machine-verified |
|---|---|---|
| fama1970 | title, author, journal, vol 25(2), start page 383 (Crossref) | end page 417 (Crossref lists only the start page) |
| henderson2018 | title, authors, AAAI vol 32(1), DOI (Crossref) | page numbers — omitted from the entry |
| li2018deeper | title, authors, AAAI vol 32(1), DOI (Crossref) | page numbers — omitted from the entry |
| vaswani2017attention | title, 8 authors in order (arXiv 1706.03762) | venue "NIPS 30, 2017" (proceedings site blocked) — pages omitted |
| melis2018 | title, authors in order (arXiv 1707.05589) | venue "ICLR 2018" (OpenReview blocked) |
| velickovic2018gat | — | entire entry (arXiv/OpenReview blocked); standard citation used |

## Corrections made to the Phase-1 review slide references
- Slide [5] "M. Llesmani and M. Michael, Temporal GNNs for policy impact on finance,
  arXiv:2404.14757" — **removed**. That arXiv ID is a different paper (Xu et al.,
  "SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting").
- Slide [13] LED-GNN (FinNLP workshop) — **removed**. The slide gives only the ACL
  Anthology homepage, and no matching record was found in Crossref.
- Slides [6] and [14] — **duplicates** of MSGformer; cited once.
- Slide [16] FinReport — corrected to *Companion* Proc. ACM Web Conf. 2024, pp. 319-327.
- Slide [21] — title completed: "...External Knowledge Infusion: Datasets and Benchmarks for Stock Markets".
- Slide [22] — added vol. 17, no. 1, art. 267.
- Slide [23] FinGraphFL — two authors (Z. Xia, S. C. Saha), not "et al.".
