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
    box(ax, 0.05, 0.25, 1.75, 0.80, "Stock graph\nsector, correlation,\ntweet co-mention", edge=BLUE, face=ours_face)
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
        ("MMGT w/o ranking term", d[d.tag == "F1c"].test_mcc.to_numpy(), BLUE),
        ("MMGT + graph-biased attention", d[d.tag == "G2c"].test_mcc.to_numpy(), BLUE),
        ("MMGT-base (price only)", d[d.tag == "F0_dtml"].test_mcc.to_numpy(), ORANGE),
        ("MMGT-base, 7 features", d[d.tag == "S4_ret_trend"].test_mcc.to_numpy(), ORANGE),
        ("MMGT-base w/o attention", d[d.tag == "F4_nosa"].test_mcc.to_numpy(), ORANGE),
        ("RGCN graph (earlier work)", e6, GREY),
        ("MLP", d[d.tag == "B_mlp"].test_mcc.to_numpy(), GREY),
        ("Attention LSTM", d[d.tag == "B_alstm"].test_mcc.to_numpy(), GREY),
        ("Logistic regression", d[d.tag == "B_lr"].test_mcc.to_numpy(), GREY),
        ("LSTM", d[d.tag == "B_lstm"].test_mcc.to_numpy(), GREY),
        ("Tweet-count LSTM (earlier work)", exp1, GREY),
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


def _all_results():
    frames = [pd.read_csv(os.path.join(R, f)) for f in
              ["final.csv", "plus_final.csv", "graph_final.csv", "baselines_final.csv",
               "subset_final.csv", "federated_final.csv"]]
    return pd.concat(frames, ignore_index=True)


def fig_timing_selection():
    """The paper's central idea: x = market timing, y = stock selection."""
    d = _all_results()
    groups = [  # tag, label, colour, filled, label offset (points)
        ("B_lstm", "LSTM", GREY, True, (6, 4)),
        ("B_alstm", "ALSTM", GREY, True, (6, -9)),
        ("B_mlp", "MLP", GREY, True, (-8, -11)),
        ("F0_dtml", "MMGT-base", ORANGE, True, (6, -9)),
        ("F1c", "+ profile", BLUE, False, (-30, 9)),
        ("F2c", "MMGT (ours)", BLUE, True, (-14, 9)),
        ("FEDc", "MMGT federated", BLUE, False, (-22, -15)),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    for tag, label, col, filled, off in groups:
        x = d[d.tag == tag]
        ax.scatter(x.test_collapse_mcc, x.test_within_day_mcc, s=8, color=col, alpha=0.25,
                   linewidths=0, zorder=2)
        mx, my = x.test_collapse_mcc.mean(), x.test_within_day_mcc.mean()
        sx = x.test_collapse_mcc.std() / np.sqrt(len(x))
        sy = x.test_within_day_mcc.std() / np.sqrt(len(x))
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="none", ecolor=col, elinewidth=0.9, zorder=3)
        ax.scatter([mx], [my], s=34, zorder=4, linewidths=1.3, edgecolors=col,
                   facecolors=col if filled else "white")
        ax.annotate(label, (mx, my), xytext=off, textcoords="offset points",
                    fontsize=6.6, color=INK)
    ax.set_xlabel("Market timing: day-collapse MCC")
    ax.set_ylabel("Stock selection: within-day MCC")
    ax.set_xlim(-0.1, 0.32)
    ax.set_ylim(-0.03, 0.09)
    ax.grid(color="#e4e3df", linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "timing_selection.pdf"))
    fig.savefig(os.path.join(OUT, "timing_selection.png"), dpi=200)
    plt.close(fig)


def fig_federated():
    """FedAvg over nine sector clients."""
    sectors = pd.read_csv(os.path.join(ROOT, "dataset", "final", "sector_mapping.csv"))
    counts = sectors.Sector.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    ax.set_xlim(0, 9.4); ax.set_ylim(0, 3.2); ax.axis("off")
    box(ax, 2.9, 2.25, 3.6, 0.82,
        "Server: weighted FedAvg\n$w \\leftarrow \\sum_k \\frac{n_k}{n}\\, w_k$", bold=True)
    short = {"Basic_Materials": "Basic\nmat.", "Conglomerates": "Conglo-\nmerates",
             "Consumer_Goods": "Cons.\ngoods", "Financial": "Finan-\ncial",
             "Healthcare": "Health-\ncare", "Industrial_Goods": "Indus.\ngoods",
             "Services": "Ser-\nvices", "Technology": "Tech-\nnology",
             "Utilities": "Utili-\nties"}
    w = 0.98
    for i, (name, c) in enumerate(counts.items()):
        x = 0.1 + i * (w + 0.04)
        ax.add_patch(FancyBboxPatch((x, 0.15), w, 0.95,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=1.0, edgecolor=BLUE, facecolor="#e8f1fc"))
        ax.text(x + w / 2, 0.62, f"{short[name]}\n({c})", ha="center", va="center",
                fontsize=5.6, color=INK, linespacing=1.15)
        ax.add_patch(FancyArrowPatch((x + w / 2, 1.12), (4.7, 2.22), arrowstyle="<|-|>",
                                     mutation_scale=6, linewidth=0.6, color=INK2, alpha=0.7))
    ax.text(4.7, 1.62, "local training on own stocks and labels;\nweights up, global model down",
            ha="center", va="center", fontsize=6.2, color=INK2,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, "federated.pdf"))
    fig.savefig(os.path.join(OUT, "federated.png"), dpi=200)
    plt.close(fig)


def fig_timeline():
    """Chronological splits."""
    spans = [("Train", "2014-01-02", "2015-07-31", "398 days, 20,309 samples", GREY),
             ("Val.", "2015-08-03", "2015-09-30", "val.: 42 days, 2,555", ORANGE),
             ("Test", "2015-10-01", "2015-12-31", "test: 64 days, 3,720", BLUE)]
    fig, ax = plt.subplots(figsize=(3.5, 1.05))
    for name, a, b, note, col in spans:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ax.barh(0, (b - a).days, left=a, height=0.5, color=col, alpha=0.85,
                edgecolor="white", linewidth=1.5)
        mid = a + (b - a) / 2
        ax.text(mid, 0, name, ha="center", va="center", fontsize=7, color="white",
                weight="bold")
        if name == "Train":
            ax.text(mid, -0.48, note, ha="center", va="top", fontsize=6, color=INK2)
        else:
            ax.text(b, -0.40 if name == "Val." else -0.72, note, ha="right", va="top",
                    fontsize=6, color=INK2)
    ax.set_ylim(-1.15, 0.45)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.set_xticks([pd.Timestamp(x) for x in ["2014-01-01", "2014-07-01", "2015-01-01",
                                             "2015-07-01", "2016-01-01"]])
    ax.tick_params(axis="x", labelsize=6.4)
    fig.tight_layout(pad=0.15)
    fig.savefig(os.path.join(OUT, "timeline.pdf"))
    fig.savefig(os.path.join(OUT, "timeline.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import matplotlib.dates  # noqa: F401  (date axis for the timeline)
    os.makedirs(OUT, exist_ok=True)
    fig_architecture()
    fig_seeds()
    fig_timing_selection()
    fig_federated()
    fig_timeline()
    print("figures written to", OUT)
