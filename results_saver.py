"""
results_saver.py
================
Saves all metrics, configs and per-sector breakdowns to disk in the
format required by the MMGTFFF roadmap.

Output structure (under results/phase1_baselines/logistic_regression/)
-----------------------------------------------------------------------
    metrics.json          — all numerical results for all 4 steps
    config.yaml           — hyperparameters and split definitions
    summary_table.csv     — clean CSV for easy copy-paste into the paper
    per_sector/
        step1_price_only.csv
        step2_price_sentiment.csv
        step3_price_fundamentals.csv
        step4_full_structured.csv

Usage
-----
    from results_saver import save_all
    save_all(results, output_dir="results/phase1_baselines/logistic_regression")
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

from data_loader import SPLITS

_LR_PARAMS = dict(C=0.1, max_iter=2000, solver="lbfgs", random_state=42)


def save_all(results: list, output_dir: str = "results/phase1_baselines/logistic_regression") -> None:
    """
    Persist all results to disk.

    Parameters
    ----------
    results    : list of result dicts from trainer.run_all_steps()
    output_dir : root directory for this model's outputs
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_sector").mkdir(exist_ok=True)

    _save_metrics_json(results, out)
    _save_config_yaml(results, out)
    _save_summary_csv(results, out)
    _save_per_sector_csvs(results, out)

    print(f"\n[results_saver] All results saved to: {out.resolve()}")
    print(f"[results_saver] Files written:")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(out)}")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _save_metrics_json(results: list, out: Path) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "steps": [],
    }
    for r in results:
        payload["steps"].append({
            "name":             r["name"],
            "feature_key":      r["feature_key"],
            "n_features":       r["n_features"],
            "accuracy":         round(r["accuracy"],  4),
            "f1_macro":         round(r["f1_macro"],  4),
            "mcc":              round(r["mcc"],        4),
            "auc":              round(r["auc"],        4),
            "confusion_matrix": r["confusion_matrix"],
        })
    path = out / "metrics.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[results_saver] Saved metrics.json")


def _save_config_yaml(results: list, out: Path) -> None:
    config = {
        "model":      "LogisticRegression",
        "params":     _LR_PARAMS,
        "splits":     {k: {"start": v[0], "end": v[1]} for k, v in SPLITS.items()},
        "leakage_fix": (
            "All same-day technical indicators and tweet counts are lagged by 1 "
            "trading day per ticker. Return = sign(Return) == Target, so using "
            "Return directly gives 100% accuracy (data leakage)."
        ),
        "fundamental_strategy": (
            "Forward-fill within ticker (filing data valid until next filing), "
            "binary presence mask added per column, zero-fill residual NaN."
        ),
        "normalisation": "StandardScaler fitted on train split only, applied to test.",
        "steps": [
            {"step": i + 1, "name": r["name"], "feature_key": r["feature_key"], "n_features": r["n_features"]}
            for i, r in enumerate(results)
        ],
    }
    path = out / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[results_saver] Saved config.yaml")


def _save_summary_csv(results: list, out: Path) -> None:
    rows = []
    for r in results:
        rows.append({
            "Step":       r["name"],
            "N_Features": r["n_features"],
            "Accuracy":   round(r["accuracy"], 4),
            "F1_Macro":   round(r["f1_macro"], 4),
            "MCC":        round(r["mcc"],       4),
            "AUC_ROC":    round(r["auc"],       4),
        })
    df = pd.DataFrame(rows)
    path = out / "summary_table.csv"
    df.to_csv(path, index=False)
    print(f"[results_saver] Saved summary_table.csv")
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


def _save_per_sector_csvs(results: list, out: Path) -> None:
    slug_map = {
        "price":            "step1_price_only",
        "price_sentiment":  "step2_price_sentiment",
        "price_fund":       "step3_price_fundamentals",
        "full":             "step4_full_structured",
    }
    for r in results:
        slug = slug_map.get(r["feature_key"], r["feature_key"])
        path = out / "per_sector" / f"{slug}.csv"
        r["per_sector"].to_csv(path, index=False)
    print(f"[results_saver] Saved 4 per-sector CSV files")
