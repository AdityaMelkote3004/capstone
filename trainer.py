"""
trainer.py
==========
Trains and evaluates Logistic Regression models for all 4 feature configurations.

Steps
-----
  Step 1 — Price Only        (15 lagged technical features)
  Step 2 — Price + Sentiment (+ 3 lagged tweet-count features)
  Step 3 — Price + Fundamentals (+ 28 EDGAR features with masks)
  Step 4 — Full Structured   (all of the above)

For each step, computes:
  - Overall:     Accuracy, F1 (macro), MCC, AUC-ROC
  - Per-sector:  Accuracy and F1
  - Confusion matrix

Usage
-----
    from trainer import run_all_steps
    results = run_all_steps(df, feat_groups)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from data_loader import get_xy, get_split_mask, TARGET


# ── Config ─────────────────────────────────────────────────────────────────────

STEPS = [
    ("Step 1 — Price Only",                    "price"),
    ("Step 2 — Price + Sentiment",             "price_sentiment"),
    ("Step 3 — Price + Fundamentals",          "price_fund"),
    ("Step 4 — Full Structured (All Modalities)", "full"),
]

LR_PARAMS = dict(
    C=0.1,           # mild L2 regularisation — avoids overfitting a ~10k train set
    max_iter=2000,
    solver="lbfgs",
    random_state=42,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_all_steps(df: pd.DataFrame, feat_groups: dict) -> list:
    """
    Train and evaluate all 4 LR steps.

    Parameters
    ----------
    df          : preprocessed DataFrame from data_loader.load_data()
    feat_groups : dict from data_loader.get_feature_groups()

    Returns
    -------
    List of result dicts, one per step. Each dict contains:
        name, feature_key, n_features, accuracy, f1_macro, mcc, auc,
        confusion_matrix (2×2 list), per_sector (DataFrame)
    """
    results = []
    for step_name, feat_key in STEPS:
        features = feat_groups[feat_key]
        print(f"\n{'-'*60}")
        print(f"  Running: {step_name}  ({len(features)} features)")
        print(f"{'-'*60}")
        r = _train_evaluate(step_name, feat_key, features, df)
        results.append(r)
        _print_result(r)
    return results


def _train_evaluate(
    name: str,
    feat_key: str,
    features: list,
    df: pd.DataFrame,
) -> dict:
    """Train on train split, evaluate on test split, compute all metrics."""

    # ── Fit scaler on train, apply to test ──────────────────────────────────
    scaler = StandardScaler()
    X_train, y_train, scaler = get_xy(df, features, "train", scaler, fit_scaler=True)
    X_test,  y_test,  _      = get_xy(df, features, "test",  scaler, fit_scaler=False)

    # ── Train ────────────────────────────────────────────────────────────────
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)

    # ── Predict ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Overall metrics ──────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)

    # ── Per-sector metrics ────────────────────────────────────────────────────
    per_sector = _per_sector_metrics(df, y_pred, y_prob, features, scaler)

    return {
        "name":             name,
        "feature_key":      feat_key,
        "n_features":       len(features),
        "accuracy":         acc,
        "f1_macro":         f1,
        "mcc":              mcc,
        "auc":              auc,
        "confusion_matrix": cm.tolist(),
        "per_sector":       per_sector,
        "model":            model,
        "scaler":           scaler,
        "features":         features,
    }


def _per_sector_metrics(
    df: pd.DataFrame,
    y_pred_all: np.ndarray,
    y_prob_all: np.ndarray,
    features: list,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Compute accuracy and F1 broken down by sector on the test split."""
    test_mask = get_split_mask(df, "test")
    test_df   = df[test_mask].copy().reset_index(drop=True)
    test_df["y_pred"] = y_pred_all
    test_df["y_prob"] = y_prob_all

    rows = []
    for sector, grp in test_df.groupby("Sector"):
        if len(grp) < 10:          # skip tiny slices
            continue
        yt = grp[TARGET].values
        yp = grp["y_pred"].values
        yb = grp["y_prob"].values
        rows.append({
            "Sector":   sector,
            "N":        len(grp),
            "Accuracy": round(accuracy_score(yt, yp), 4),
            "F1_macro": round(f1_score(yt, yp, average="macro"), 4),
            "AUC":      round(roc_auc_score(yt, yb), 4) if len(np.unique(yt)) > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def _print_result(r: dict) -> None:
    print(f"\n  Accuracy   : {r['accuracy']:.4f}")
    print(f"  F1 (macro) : {r['f1_macro']:.4f}")
    print(f"  MCC        : {r['mcc']:.4f}")
    print(f"  AUC-ROC    : {r['auc']:.4f}")
    cm = np.array(r["confusion_matrix"])
    print(f"  Confusion Matrix:\n"
          f"              Pred 0   Pred 1\n"
          f"    True 0  [ {cm[0,0]:5d}   {cm[0,1]:5d} ]\n"
          f"    True 1  [ {cm[1,0]:5d}   {cm[1,1]:5d} ]\n")
    print("  Per-Sector Breakdown:")
    print(r["per_sector"].to_string(index=False))
