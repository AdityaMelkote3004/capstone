"""
visualiser.py
=============
Generates all plots for the baseline LR pipeline.

Plots produced
--------------
1.  metrics_comparison.png  — grouped bar chart (Acc / F1 / MCC / AUC) across 4 steps
2.  confusion_matrices.png  — 1×4 heatmap grid
3.  per_sector_heatmap.png  — AUC by (Step × Sector) heatmap
4.  feature_importance.png  — top-20 LR coefficients for the Full model (Step 4)
5.  class_distribution.png  — target balance across train / test splits

Usage
-----
    from visualiser import plot_all
    plot_all(results, df, output_dir="results/phase1_baselines/logistic_regression")
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Colour palette ─────────────────────────────────────────────────────────────
STEP_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
METRIC_COLOURS = {
    "Accuracy": "#4C72B0",
    "F1 (macro)": "#DD8452",
    "MCC": "#55A868",
    "AUC-ROC": "#C44E52",
}
STEP_SHORT = [
    "Price\nOnly",
    "Price +\nSentiment",
    "Price +\nFundamentals",
    "Full\nStructured",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_all(results: list, df: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Generate and save all plots.

    Parameters
    ----------
    results    : list of result dicts from trainer.run_all_steps()
    df         : preprocessed DataFrame from data_loader.load_data()
    output_dir : directory to save all PNG files
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(results, out)
    plot_confusion_matrices(results, out)
    plot_per_sector_heatmap(results, out)
    plot_feature_importance(results, out)
    plot_class_distribution(df, out)

    print(f"\n[visualiser] All plots saved to: {out.resolve()}")


def plot_metrics_comparison(results: list, out: Path) -> None:
    """Grouped bar chart of all 4 metrics across all 4 steps."""
    metrics    = ["accuracy", "f1_macro", "mcc", "auc"]
    labels     = ["Accuracy", "F1 (macro)", "MCC", "AUC-ROC"]
    n_steps    = len(results)
    n_metrics  = len(metrics)
    x          = np.arange(n_steps)
    width      = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [r[metric] for r in results]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=list(METRIC_COLOURS.values())[i], alpha=0.87, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(STEP_SHORT, fontsize=10)
    ax.set_ylim(0.35, 0.68)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Logistic Regression Baselines — Stepwise Modality Addition\n"
        "Train: Jan–Oct 2014  |  Test: Nov–Dec 2015",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = out / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualiser] Saved {path.name}")


def plot_confusion_matrices(results: list, out: Path) -> None:
    """1×4 grid of confusion matrix heatmaps."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle(
        "Confusion Matrices  (0 = Down, 1 = Up)\nTest set: Nov–Dec 2015",
        fontsize=12, fontweight="bold",
    )

    for ax, r, colour in zip(axes, results, STEP_COLOURS):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred ↓", "Pred ↑"],
            yticklabels=["True ↓", "True ↑"],
            cbar=False, annot_kws={"size": 13, "weight": "bold"},
            linewidths=0.5, linecolor="white",
        )
        total   = cm.sum()
        correct = np.trace(cm)
        ax.set_title(
            f"{STEP_SHORT[results.index(r)]}\n"
            f"Acc: {correct/total:.3f}",
            fontsize=9, fontweight="bold", pad=8,
        )

    fig.tight_layout()
    path = out / "confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualiser] Saved {path.name}")


def plot_per_sector_heatmap(results: list, out: Path) -> None:
    """AUC heatmap: rows = sectors, columns = steps."""
    # Build matrix
    all_sectors = sorted(
        set(s for r in results for s in r["per_sector"]["Sector"].tolist())
    )
    step_labels = [f"S{i+1}" for i in range(len(results))]
    matrix      = pd.DataFrame(index=all_sectors, columns=step_labels, dtype=float)

    for i, r in enumerate(results):
        for _, row in r["per_sector"].iterrows():
            matrix.loc[row["Sector"], f"S{i+1}"] = row["AUC"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix.astype(float),
        annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.44, vmax=0.60,
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 10},
        cbar_kws={"label": "AUC-ROC"},
    )
    ax.set_title(
        "AUC-ROC by Sector and Step\n"
        "S1=Price  S2=Price+Sent  S3=Price+Fund  S4=Full",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Sector", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    path = out / "per_sector_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualiser] Saved {path.name}")


def plot_feature_importance(results: list, out: Path) -> None:
    """
    Top-20 absolute LR coefficients for the Full model (Step 4).
    Positive coef = feature pushes towards predicting UP.
    """
    full_result = results[-1]   # Step 4 — Full Structured
    model       = full_result["model"]
    features    = full_result["features"]

    coefs = model.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[::-1][:20]

    top_features = [features[i] for i in top_idx]
    top_coefs    = coefs[top_idx]

    # Clean up feature names for display
    display_names = [f.replace("_lag1", " (lag1)").replace("_present", " mask") for f in top_features]

    colours = ["#C44E52" if c > 0 else "#4C72B0" for c in top_coefs]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_coefs, color=colours, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LR Coefficient (standardised features)", fontsize=10)
    ax.set_title(
        "Top 20 Feature Importances — Full Structured Model (Step 4)\n"
        "Red = predicts UP  |  Blue = predicts DOWN",
        fontsize=11, fontweight="bold",
    )

    up_patch   = mpatches.Patch(color="#C44E52", label="Pushes toward UP (1)")
    down_patch = mpatches.Patch(color="#4C72B0", label="Pushes toward DOWN (0)")
    ax.legend(handles=[up_patch, down_patch], fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = out / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualiser] Saved {path.name}")


def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of class balance across train and test splits."""
    from data_loader import get_split_mask, TARGET

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Target Class Distribution (0=Down, 1=Up)", fontsize=12, fontweight="bold")

    split_configs = [
        ("train", "Train  (Jan–Oct 2014)", "#4C72B0"),
        ("test",  "Test   (Nov–Dec 2015)", "#C44E52"),
    ]

    for ax, (split, label, colour) in zip(axes, split_configs):
        mask   = get_split_mask(df, split)
        counts = df.loc[mask, TARGET].value_counts().sort_index()
        bars = []
        for j, (lbl, val) in enumerate(zip(["Down (0)", "Up (1)"], counts.values)):
            alpha = 0.6 if j == 0 else 0.9
            b = ax.bar(lbl, val, color=colour, alpha=alpha, edgecolor="white", width=0.5)
            bars.append(b[0])
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20,
                    f"{val}\n({val/counts.sum()*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=9)
        ax.set_ylim(0, counts.max() * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = out / "class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualiser] Saved {path.name}")
