import numpy as np

from src.utils.metrics import (market_direction_oracle, pooled_metrics,
                               within_day_metrics)


def test_pooled_metrics_on_a_perfect_predictor():
    targets = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    m = pooled_metrics(targets, probs)
    assert m["accuracy"] == 1.0
    assert m["mcc"] == 1.0
    assert m["auc"] == 1.0


def test_pooled_metrics_single_class_auc_falls_back_to_half():
    targets = np.array([1, 1, 1])
    probs = np.array([0.6, 0.7, 0.8])
    assert pooled_metrics(targets, probs)["auc"] == 0.5


def test_within_day_skips_single_class_days_and_counts_them():
    targets = np.array([1, 1, 0, 1])
    probs = np.array([0.9, 0.8, 0.2, 0.7])
    days = np.array(["d1", "d1", "d2", "d2"])
    m = within_day_metrics(targets, probs, days)
    assert m["days_scored"] == 1, "d1 is single-class and must be skipped"
    assert m["days_skipped"] == 1


def test_within_day_mcc_is_perfect_when_each_day_is_ranked_right():
    targets = np.array([0, 1, 0, 1])
    probs = np.array([0.2, 0.8, 0.3, 0.9])
    days = np.array(["d1", "d1", "d2", "d2"])
    assert within_day_metrics(targets, probs, days)["within_day_mcc"] == 1.0


def test_within_day_exposes_a_market_direction_only_predictor():
    """THE point of this module: a predictor that only knows the day's
    direction scores high pooled but zero within-day."""
    targets = np.array([1, 1, 1, 0, 0, 0])
    days = np.array(["d1", "d1", "d1", "d2", "d2", "d2"])
    probs = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])   # identical within each day
    assert pooled_metrics(targets, probs)["accuracy"] == 1.0
    m = within_day_metrics(targets, probs, days)
    assert m["days_scored"] == 0, "no day has both classes, so nothing is scored"


def test_within_day_zero_when_predictions_are_flat_within_mixed_days():
    targets = np.array([0, 1, 0, 1])
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    days = np.array(["d1", "d1", "d2", "d2"])
    assert within_day_metrics(targets, probs, days)["within_day_mcc"] == 0.0


def test_market_direction_oracle_matches_the_measured_test_value():
    targets = np.array([1, 1, 1, 0, 0, 1])
    days = np.array(["d1", "d1", "d1", "d2", "d2", "d2"])
    m = market_direction_oracle(targets, days)
    # d1 majority UP (3/3 correct); d2 majority DOWN (2/3 correct)
    assert m["accuracy"] == 5 / 6
