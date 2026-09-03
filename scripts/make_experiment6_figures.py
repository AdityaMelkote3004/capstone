"""Generates Experiment 6's figures from the edge cache and results table."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.data.edge_cache import load_edge_cache

OUT_DIR = "results/experiment6_dense_graph"
FIG_DIR = os.path.join(OUT_DIR, "figures")
SECTOR_CSV = "dataset/final/sector_mapping.csv"
BASELINES = {"Exp1 LSTM+tweets": 0.0922, "Exp3 price-only": 0.0108, "Exp4 sector GAT": 0.0278}


def draw_graph(pairs, sectors, title, path):
    g = nx.Graph()
    g.add_nodes_from(sectors)
    g.add_edges_from(pairs)
    codes = sorted(set(sectors.values()))
    colors = [codes.index(sectors[n]) for n in g.nodes()]
    plt.figure(figsize=(9, 9))
    nx.draw_networkx(
        g, pos=nx.spring_layout(g, seed=42, k=0.35), node_color=colors,
        cmap="tab10", node_size=220, font_size=6, edge_color="#bbbbbb", width=0.5,
    )
    plt.title(f"{title}\n{g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sectors = pd.read_csv(SECTOR_CSV).set_index("Ticker")["Sector"].to_dict()
    cache = load_edge_cache(os.path.join(OUT_DIR, "edge_cache.pkl"))
    day = cache[sorted(cache)[300]]     # a mid-dataset day, past correlation warm-up

    for rel, title in [("sector", "Sector edges"), ("correlation", "Correlation edges (60d, top-5)"),
                       ("comention", "Tweet co-mention edges")]:
        draw_graph(day[rel], sectors, title, os.path.join(FIG_DIR, f"graph_{rel}.png"))

    # Edge density over time
    dates = sorted(cache)
    plt.figure(figsize=(12, 4))
    for rel in ("sector", "correlation", "comention"):
        plt.plot(range(len(dates)), [len(cache[d][rel]) for d in dates], label=rel, linewidth=1)
    plt.xlabel("trading day index"); plt.ylabel("edges")
    plt.title("Edge count per trading day, by relation")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "edge_density_over_time.png"), dpi=150)
    plt.close()

    results_csv = os.path.join(OUT_DIR, "results.csv")
    if not os.path.exists(results_csv):
        print("No results.csv yet; skipping result figures.")
        return
    df = pd.read_csv(results_csv)

    # Mean MCC per (model, edge combo), with seed spread as error bars
    agg = df.groupby(["model", "edges"])["mcc"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(13, 6))
    models = sorted(agg["model"].unique())
    combos = sorted(agg["edges"].unique())
    width = 0.8 / max(len(models), 1)
    for mi, m in enumerate(models):
        sub = agg[agg["model"] == m].set_index("edges").reindex(combos)
        xs = [i + mi * width for i in range(len(combos))]
        plt.bar(xs, sub["mean"].fillna(0), width=width,
                yerr=sub["std"].fillna(0), capsize=2, label=m)
    for name, val in BASELINES.items():
        plt.axhline(val, linestyle="--", linewidth=1, alpha=0.7)
        plt.text(len(combos) - 0.5, val, f" {name} ({val:+.4f})", va="bottom", fontsize=7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks([i + 0.4 for i in range(len(combos))], combos, rotation=20, ha="right", fontsize=8)
    plt.ylabel("test MCC (mean over seeds)")
    plt.title("Experiment 6: test MCC by model and edge combination")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mcc_by_model_and_edges.png"), dpi=150)
    plt.close()

    # Seed spread, so single-seed noise is visible rather than implied
    if df["seed"].nunique() > 1:
        plt.figure(figsize=(13, 5))
        labels, spreads = [], []
        for (m, e), grp in df.groupby(["model", "edges"]):
            labels.append(f"{m}\n{e}")
            spreads.append(grp["mcc"].max() - grp["mcc"].min())
        plt.bar(range(len(labels)), spreads)
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.ylabel("max - min test MCC across seeds")
        plt.title("Seed spread per grid cell (differences smaller than this are noise)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "seed_variance.png"), dpi=150)
        plt.close()

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
