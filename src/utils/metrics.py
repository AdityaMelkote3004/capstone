"""Pooled and within-day metrics for Experiment 7.

Measured on the protocol-A splits, 48.3% of test-set label variance is
BETWEEN-day: on a given trading day the ~58 tickers in the cross-section mostly
move together, and 32 of 64 test days are more than 80% one-directional. An
oracle knowing only each day's market direction -- with no ability to tell
stocks apart -- scores pooled MCC +0.6248 and accuracy 0.8116, far above every
published number on this benchmark.

Pooled MCC therefore rewards two different abilities at once. The graph exists
to discriminate between stocks, so every result must be reported with within-day
metrics beside the pooled ones.

This is not leakage: market direction is legitimately part of the task as
StockNet defines it, and all published numbers are pooled. It is a
measurement-validity issue.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)


def pooled_metrics(targets, probs) -> dict:
    """Standard metrics over all samples, matching Experiments 1/3/4/5/6."""
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, preds)),
        "auc": float(roc_auc_score(targets, probs)) if len(set(targets.tolist())) > 1 else 0.5,
    }


def within_day_metrics(targets, probs, days) -> dict:
    """MCC and accuracy computed inside each day's cross-section, then averaged.

    Days with a single class present are skipped (MCC is undefined there) and
    counted, so the reader can see how much of the test set the within-day
    number actually covers.

    Predictions are compared against each day's own median probability rather
    than a fixed 0.5, so a model that shifts every ticker on a day in the same
    direction gains nothing here -- which is exactly what separates stock
    discrimination from market-direction prediction.
    """
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    days = np.asarray(days)

    mccs, accuracies, skipped = [], [], 0
    for day in np.unique(days):
        mask = days == day
        day_targets = targets[mask]
        if len(set(day_targets.tolist())) < 2:
            skipped += 1
            continue
        day_probs = probs[mask]
        day_preds = (day_probs >= np.median(day_probs)).astype(int)
        mccs.append(matthews_corrcoef(day_targets, day_preds))
        accuracies.append(accuracy_score(day_targets, day_preds))

    return {
        "within_day_mcc": float(np.mean(mccs)) if mccs else 0.0,
        "within_day_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "days_scored": len(mccs),
        "days_skipped": skipped,
    }


def market_direction_oracle(targets, days) -> dict:
    """Upper bound for a predictor that knows each day's majority direction and
    nothing else. Reported as a reference row so readers can see what pooled
    MCC is capable of rewarding without any stock discrimination at all."""
    targets = np.asarray(targets)
    days = np.asarray(days)
    preds = np.zeros_like(targets)
    for day in np.unique(days):
        mask = days == day
        preds[mask] = int(targets[mask].mean() >= 0.5)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "mcc": float(matthews_corrcoef(targets, preds)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
    }
