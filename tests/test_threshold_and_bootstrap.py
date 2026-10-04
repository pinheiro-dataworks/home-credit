"""Threshold selection (F1 and cost-based) and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np

from src.utils.stats import bootstrap_metrics, find_optimal_threshold


def _synthetic_scores(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.choice([0, 1], n, p=[0.9, 0.1])
    # Positives score systematically higher than negatives, with noise.
    y_prob = np.clip(
        rng.normal(0.25, 0.15, n) + y_true * rng.normal(0.35, 0.15, n), 0, 1
    )
    return y_true, y_prob


def test_find_optimal_threshold_f1_in_valid_range():
    y_true, y_prob = _synthetic_scores()
    thr, val = find_optimal_threshold(y_true, y_prob, metric="f1", n_thresholds=100)

    assert 0.01 <= thr <= 0.99
    assert 0.0 <= val <= 1.0


def test_cost_threshold_moves_down_as_false_negative_cost_rises():
    """Minimising cost_fn * FN + cost_fp * FP with an expensive false negative
    should select a lower (more permissive) threshold than treating both error
    types equally, since flagging more applicants as risky reduces missed
    defaults at the expense of more false positives."""
    y_true, y_prob = _synthetic_scores()

    thr_balanced, _ = find_optimal_threshold(
        y_true, y_prob, metric="cost", n_thresholds=100, cost_fn=1.0, cost_fp=1.0
    )
    thr_fn_expensive, _ = find_optimal_threshold(
        y_true, y_prob, metric="cost", n_thresholds=100, cost_fn=20.0, cost_fp=1.0
    )

    assert thr_fn_expensive <= thr_balanced


def test_bootstrap_metrics_ci_is_sane():
    y_true, y_prob = _synthetic_scores(n=800)
    ci = bootstrap_metrics(y_true, y_prob, threshold=0.5, n_iterations=200, ci=0.95, random_state=1)

    for metric in ("auc_roc", "auc_pr", "precision", "recall", "f1"):
        entry = ci[metric]
        assert entry["lower"] <= entry["point"] <= entry["upper"]
        assert 0.0 <= entry["lower"] and entry["upper"] <= 1.0
