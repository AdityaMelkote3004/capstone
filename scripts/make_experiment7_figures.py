"""Experiment 7 figures: the ablation ladder, pooled vs within-day, and the
FiLM explainability readout.

The gamma/beta readout is a deliverable independent of the MCC outcome -- if
the ladder shows no gain, "macro events measurably change how per-ticker text
is interpreted" is still a reportable finding.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.macro_encoder import load_macro_calendar

OUT_DIR = "results/experiment7_macro_conditioned"
FIG_DIR = os.path.join(OUT_DIR, "figures")
REFERENCES = {
    "Exp1 LSTM+tweets": 0.0922,
    "Exp6 best (RGCN sec+corr)": 0.0677,
    "Exp6 GAT sector (3-seed)": 0.0160,
}


def ladder_figure(df: pd.DataFrame) -> None:
    agg = df.groupby("row").agg(
        pooled=("mcc", "mean"), pooled_std=("mcc", "std"),
        within=("within_day_mcc", "mean"), within_std=("within_day_mcc", "std"),
    ).reset_index().sort_values("row")

    x = range(len(agg))
    width = 0.38
    plt.figure(figsize=(13, 6))
    plt.bar([i - width / 2 for i in x], agg["pooled"], width,
            yerr=agg["pooled_std"].fillna(0), capsize=3, label="pooled MCC")
    plt.bar([i + width / 2 for i in x], agg["within"], width,
            yerr=agg["within_std"].fillna(0), capsize=3, label="within-day MCC")
    for name, value in REFERENCES.items():
        plt.axhline(value, linestyle="--", linewidth=1, alpha=0.7)
        plt.text(len(agg) - 0.4, value, f" {name} ({value:+.4f})",
                 va="bottom", fontsize=7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(list(x), agg["row"], rotation=25, ha="right", fontsize=8)
    plt.ylabel("test MCC (mean over 5 seeds)")
    plt.title("Experiment 7 ablation ladder: pooled vs within-day MCC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_ladder.png"), dpi=150)
    plt.close()


def seed_spread_figure(df: pd.DataFrame) -> None:
    # within-day, not pooled: within-day MCC is the primary metric, so the noise
    # floor quoted in the write-up must be the spread of that same quantity.
    spread = df.groupby("row")["within_day_mcc"].apply(lambda g: g.max() - g.min())
    floor = spread.mean()
    plt.figure(figsize=(11, 4))
    plt.bar(range(len(spread)), spread.values)
    plt.axhline(floor, linestyle="--", color="red", linewidth=1)
    plt.text(0, floor, f" Exp7 noise floor, mean spread ({floor:.4f})",
             va="bottom", fontsize=8, color="red")
    plt.axhline(0.0425, linestyle=":", color="green", linewidth=1.2)
    plt.text(0, 0.0425, " graph effect, row3 - row1 (+0.0425)", va="bottom",
             fontsize=8, color="green")
    plt.xticks(range(len(spread)), spread.index, rotation=25, ha="right", fontsize=8)
    plt.ylabel("max - min within-day MCC across seeds")
    plt.title("Seed spread per ladder row (differences below this are noise)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "seed_spread.png"), dpi=150)
    plt.close()


def film_figure(film_csv: str) -> None:
    """Plots gamma/beta magnitude per trading day against macro release dates.

    Expects `film_csv` with columns date,gamma_norm,beta_norm -- written by the
    runner's explainability dump (see Step 2).
    """
    if not os.path.exists(film_csv):
        print("no film_by_day.csv yet; skipping the explainability figure")
        return
    film = pd.read_csv(film_csv).sort_values("date")
    calendar = load_macro_calendar("dataset/macro_calendar.csv")

    plt.figure(figsize=(13, 4.5))
    plt.plot(range(len(film)), film["gamma_norm"], linewidth=1, label="||gamma||")
    plt.plot(range(len(film)), film["beta_norm"], linewidth=1, label="||beta||")
    for i, day in enumerate(film["date"]):
        flags = calendar.get(str(day)[:10], (0, 0, 0))
        if flags[0]:
            plt.axvline(i, color="red", alpha=0.25, linewidth=1)
    plt.xlabel("trading day index (red = FOMC)")
    plt.ylabel("FiLM parameter norm")
    plt.title("Macro conditioning strength per trading day, against FOMC dates")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "film_by_day.png"), dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    results = os.path.join(OUT_DIR, "results.csv")
    if not os.path.exists(results):
        print("no results.csv yet; run scripts/run_experiment7_ladder.py first")
        return
    df = pd.read_csv(results)
    ladder_figure(df)
    seed_spread_figure(df)
    film_figure(os.path.join(OUT_DIR, "film_by_day.csv"))
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
