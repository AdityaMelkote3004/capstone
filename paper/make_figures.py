"""Figures for the paper, generated from the committed result CSVs.

Fig. 1  architecture diagram (vector PDF)
Fig. 2  per-seed pooled test MCC for every model, baselines through ours

Colours: palette validated with the dataviz skill's validator (blue #2a78d6 /
orange #eb6834, all checks pass in light mode); baselines are neutral grey.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results", "experiment11_dtml")
OUT = os.path.join(HERE, "figures")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK})


def box(ax, x, y, w, h, text, edge=INK2, face="#ffffff", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.0, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2,
            color=INK, weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=INK2))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    ax.set_xlim(0, 10.4); ax.set_ylim(0.15, 3.2); ax.axis("off")
    ours_face = "#e8f1fc"
    # inputs
    box(ax, 0.05, 2.35, 1.75, 0.75, "Price window\n10 days $\\times$ 11\nfeatures per stock")
    box(ax, 0.05, 1.30, 1.75, 0.80, "Multimodal profile\nvolatility, SEC\nfundamentals, tweets", edge=BLUE, face=ours_face)
    box(ax, 0.05, 0.25, 1.75, 0.80, "Stock graph\nsector, correlation,\nco-mention, residual", edge=BLUE, face=ours_face)
    # encoders
    box(ax, 2.25, 2.35, 1.75, 0.75, "Attention LSTM\n+ context\nnormalisation")
    box(ax, 2.25, 1.30, 1.75, 0.80, "Profile encoder\nlinear, 47 $\\rightarrow$ 64", edge=BLUE, face=ours_face)
    box(ax, 2.25, 0.25, 1.75, 0.80, "Peer features\nneighbour means,\ncontrasts, sector", edge=BLUE, face=ours_face)
    # market context
    box(ax, 4.45, 2.35, 1.6, 0.75, "Market context\n$\\beta \\cdot h_{\\mathrm{market}}$")
    # fusion
    box(ax, 4.45, 1.05, 1.6, 0.95, "Fused stock\ncontext $h_u$\n(one per stock)", bold=True)
    # stock attention
    box(ax, 6.45, 0.80, 1.95, 1.45, "Stock-to-stock\nmulti-head\nattention\n(learned dense graph,\noptional relation bias)", bold=True)
    box(ax, 8.75, 1.05, 1.55, 0.95, "Per-stock\nprobability of\nnext-day rise")
    arrow(ax, 1.80, 2.72, 2.25, 2.72)
    arrow(ax, 1.80, 1.70, 2.25, 1.70)
    arrow(ax, 1.80, 0.65, 2.25, 0.65)
    arrow(ax, 4.00, 2.72, 4.70, 2.00)
    arrow(ax, 4.00, 1.70, 4.45, 1.55)
    arrow(ax, 4.00, 0.65, 4.70, 1.05)
    arrow(ax, 5.25, 2.35, 5.25, 2.00)
    arrow(ax, 6.05, 1.52, 6.50, 1.52)
    arrow(ax, 8.35, 1.52, 8.75, 1.52)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, "architecture.pdf"))
    fig.savefig(os.path.join(OUT, "architecture.png"), dpi=200)
    plt.close(fig)


def fig_seeds():
    frames = [pd.read_csv(os.path.join(R, f)) for f in
              ["final.csv", "plus_final.csv", "graph_final.csv", "baselines_final.csv", "subset_final.csv"]]
    d = pd.concat(frames, ignore_index=True)
    e6 = pd.read_csv(os.path.join(ROOT, "results", "experiment6_dense_graph", "results.csv"))
    e6 = e6[(e6.model == "RGCN") & (e6.edges == "sector+correlation")].mcc.to_numpy()
    exp1 = np.array([0.0027, -0.0009, 0.0011, 0.0181, -0.0125])  # Exp 1 write-up, 5-seed rerun

    rows = [  # (label, values, colour) top to bottom; only leak-free ("c") profile runs
        ("MMGT (ours)", d[d.tag == "F2c"].test_mcc.to_numpy(), BLUE),
        ("MMGT w/o ranking loss", d[d.tag == "F1c"].test_mcc.to_numpy(), BLUE),
        ("MMGT + graph-biased attention", d[d.tag == "G2c"].test_mcc.to_numpy(), BLUE),
        ("DTML (reproduced)", d[d.tag == "F0_dtml"].test_mcc.to_numpy(), ORANGE),
        ("DTML, 7 price features", d[d.tag == "S4_ret_trend"].test_mcc.to_numpy(), ORANGE),
        ("DTML w/o stock attention", d[d.tag == "F4_nosa"].test_mcc.to_numpy(), ORANGE),
        ("RGCN graph (Exp. 6)", e6, GREY),
        ("MLP", d[d.tag == "B_mlp"].test_mcc.to_numpy(), GREY),
        ("Attention LSTM", d[d.tag == "B_alstm"].test_mcc.to_numpy(), GREY),
        ("Logistic regression", d[d.tag == "B_lr"].test_mcc.to_numpy(), GREY),
        ("LSTM", d[d.tag == "B_lstm"].test_mcc.to_numpy(), GREY),
        ("Tweet-count LSTM (Exp. 1)", exp1, GREY),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    rng = np.random.default_rng(0)
    for k, (label, vals, col) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.scatter(vals, y + rng.uniform(-0.13, 0.13, len(vals)), s=9, color=col,
                   alpha=0.55, linewidths=0, zorder=2)
        m = vals.mean()
        ax.plot([m, m], [y - 0.32, y + 0.32], color=col, linewidth=2.0, zorder=3,
                solid_capstyle="round")
        ax.text(0.318, y, f"{m:+.3f}", va="center", ha="right", fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.8)
    ax.axvline(0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.04, 0.32)
    ax.set_xticks([0, 0.10, 0.20])
    ax.set_xticklabels(["0", "0.10", "0.20"])
    ax.set_xlabel("Pooled test MCC")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "seeds.pdf"))
    fig.savefig(os.path.join(OUT, "seeds.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_architecture()
    fig_seeds()
    print("figures written to", OUT)
