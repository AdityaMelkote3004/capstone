"""
run_baseline.py
===============
Entry point for Aayush's Logistic Regression Baseline pipeline.

MMGTFFF Capstone — Phase 1: Baseline Models (Member 1)
Branch: feat/data-baselines

What this script does
---------------------
1.  Loads and preprocesses stocknet_final_modeling_set.csv
2.  Builds 4 feature configurations (price → price+sentiment → price+fundamentals → full)
3.  Trains a Logistic Regression for each configuration
    - Train: Jan 2014 – Oct 2014  (first 10 months)
    - Test:  Nov 2015 – Dec 2015  (last 2 months)
4.  Evaluates: Accuracy, F1 (macro), MCC, AUC-ROC + per-sector breakdown
5.  Saves all metrics + plots to results/phase1_baselines/logistic_regression/

Leakage note
------------
`Return` and all same-day technical indicators are lagged by 1 trading day
per ticker before being used as features. `Return = sign(Return) == Target`
so using Return directly gives 100% accuracy — a well-known pitfall.

Usage
-----
    # From inside the baselineLR/ folder:
    python run_baseline.py --data ../stocknet_final_modeling_set.csv

    # Explicit output directory:
    python run_baseline.py --data /path/to/data.csv --out results/
"""

import argparse
import sys
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Allow running from the baselineLR/ folder directly ────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data_loader   import load_data, get_feature_groups
from trainer       import run_all_steps
from visualiser    import plot_all
from results_saver import save_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logistic Regression Baselines for MMGTFFF capstone"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="stocknet_final_modeling_set.csv",
        help="Path to the preprocessed CSV file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/phase1_baselines/logistic_regression",
        help="Output directory for metrics, plots and per-sector CSVs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  MMGTFFF — Phase 1: LR Baselines")
    print("  Member 1 (Aayush Prem) — feat/data-baselines")
    print("=" * 65)

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    df = load_data(args.data)

    # ── 2. Build feature groups ──────────────────────────────────────────────
    feat_groups = get_feature_groups(df)
    print(f"\n[main] Feature groups ready:")
    for k, v in feat_groups.items():
        print(f"  {k:20s}: {len(v):2d} features")

    # ── 3. Train & evaluate ──────────────────────────────────────────────────
    results = run_all_steps(df, feat_groups)

    # ── 4. Visualise ─────────────────────────────────────────────────────────
    print("\n[main] Generating plots ...")
    plot_all(results, df, output_dir=args.out)

    # ── 5. Save metrics & configs ────────────────────────────────────────────
    save_all(results, output_dir=args.out)

    print("\n[main] ✓ Baseline pipeline complete.")
    print(f"[main]   Results in: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
