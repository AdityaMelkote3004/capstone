# Related Work — Candidate Papers for Next Steps

Compiled while diagnosing Experiment 4's over-smoothing bug and deciding
what to try next (sparser edge types, better GNN training regime, and the
project's eventual event-conditioned/federated extensions). Organized by
the specific question each group of papers speaks to. Papers marked
**(verified)** were actually fetched and read (in full or via extracted
text) during this project; papers marked **(recalled)** are from general
knowledge and should be checked before citing anything specific from them
in the paper.

## 1. Why the sector graph over-smoothed, and what to do about it

Directly relevant to the bug just fixed (complete-clique graph + 2-layer
GATConv erasing per-node signal).

- **Veličković et al., "Graph Attention Networks," ICLR 2018** (arXiv:1710.10903)
  **(verified)** — the actual algorithm `torch_geometric.nn.GATConv`
  implements. Reference hyperparameters (lr 0.005, weight_decay 5e-4,
  dropout 0.6, patience 100) were pulled from here and used in the
  ablation that helped isolate the bug (though the bug turned out to be
  architectural, not a hyperparameter problem).
- **Li, Han, Wu, "Deeper Insights into Graph Convolutional Networks for
  Semi-Supervised Learning," AAAI 2018** **(recalled)** — the original
  paper characterizing GCN/GNN over-smoothing as stacking layers causes
  node representations to converge to indistinguishable values. Worth
  reading in full before citing; this project's own empirical finding
  (bit-identical outputs after 1-2 layers on a complete graph) is a
  particularly severe instance of exactly this phenomenon.
- **Oono & Suzuki, "Graph Neural Networks Exponentially Lose Expressive
  Power for Node Classification," ICLR 2020** **(recalled)** — theoretical
  treatment of why over-smoothing happens and how fast, as a function of
  graph structure. Directly relevant since our graph (complete cliques) is
  close to a worst case for this effect.
- **Rong et al., "DropEdge: Towards Deep Graph Convolutional Networks on
  Node Classification," ICLR 2020** **(recalled)** — randomly drops edges
  during training as an over-smoothing mitigation; an alternative or
  complement to the residual-connection fix already applied here.
- **Zhao & Akoglu, "PairNorm: Tackling Oversmoothing in GNNs," ICLR 2020**
  **(recalled)** — a normalization-layer-based fix for over-smoothing;
  worth comparing against the residual-connection approach if the sector
  graph (or any other dense graph construction) is revisited.
- **Brody, Alon, Yahav, "How Attentive are Graph Attention Networks?,"
  ICLR 2022** (arXiv:2105.14491) **(found via search, not yet read in
  full)** — studies a specific limitation of GAT's "static" attention
  mechanism (GATv2 is the proposed fix). Worth checking whether GATv2
  (`torch_geometric.nn.GATv2Conv`) behaves differently on our complete-clique
  graph before assuming residual connections are the only lever.

## 2. Stock-graph construction — alternatives to "same sector"

Directly relevant to the recommended next step (sparser, more selective
edges than a complete per-sector clique).

- **Kim et al., "HATS: A Hierarchical Graph Attention Network for Stock
  Movement Prediction," 2019** (arXiv:1908.07999) **(verified, abstract +
  intro read in full; training hyperparameters not published in the
  extractable text)** — uses a *multi-graph* (several relation types
  simultaneously, e.g. sector, supply chain) with learned per-relation-type
  attention weights, rather than one flat "same sector" graph. Their
  approach to combining multiple edge types with attention is a good
  reference for the project's planned correlation + dynamic edges.
- **Feng et al., "Temporal Relational Ranking for Stock Prediction," ACM
  TOIS 2019** **(recalled — this is the paper HATS's own repo links to for
  "how to generate the graph," so it's likely the source of a Wikidata- or
  sector-based relation-graph construction method used in that line of
  work)** — should be read directly rather than inferred from HATS's
  README pointer.
- **Sawhney et al., "Deep Attentive Learning for Stock Movement Prediction
  From Social Media Text and Company Correlations," EMNLP 2020**
  (aclanthology.org/2020.emnlp-main.676) **(verified — full text read)** —
  MAN-SF itself. Confirmed via direct reading: their graph is built from
  Wikidata first/second-order company relations (e.g. "owned by," "board
  member of," shared founder/industry), not sector labels or price
  correlation. This is the paper Experiment 4's docs already cite for the
  "reproduction vs. contribution" framing.
- **Chen, Wei, Huang, "Incorporating Corporation Relationship via Graph
  Convolutional Neural Networks for Stock Price Prediction," CIKM 2018**
  **(recalled)** — an early GCN-over-corporate-relations paper; useful
  point of comparison for how much the choice of GCN vs. GAT matters
  versus the choice of edge type.
- **DGDNN: "Decoupled Graph Diffusion Neural Network for Stock Movement
  Prediction," 2024** (arXiv:2401.01846) **(found via search, not read)** —
  more recent work specifically on stock-movement GNNs; worth checking
  whether it addresses over-smoothing or dense-graph pathologies directly,
  given how recent it is relative to the DropEdge/PairNorm line above.

## 3. Federated learning (the project's actual novelty axis)

MAN-SF has no federated component at all — this is where MMGTFFF's real
contribution over the literature lives, per the reproduction-vs-contribution
framing already established. Not yet researched in this project; flagging
as the next literature gap to close before building `src/training/federated.py`.

- **McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data," AISTATS 2017** **(recalled)** — the original
  FedAvg paper. Should be read directly before implementing FedAvg, not
  assumed from memory.
- **Li et al., "Federated Optimization in Heterogeneous Networks," MLSys
  2020** **(recalled)** — introduces FedProx (the proximal-term variant
  CLAUDE.md's roadmap already names as a target). Directly relevant to the
  planned `FederatedTrainer` class.

## What to actually do with this list

Before the next architecture change (sparser edge types) or the federated
phase, at minimum read Li et al. 2018 (over-smoothing) and Feng et al. 2019
(the relation-graph construction HATS points to) in full — both are marked
**(recalled)** above, meaning nothing specific has been verified from them
yet, unlike the MAN-SF/GAT/HATS papers this project has already fetched and
checked directly.
