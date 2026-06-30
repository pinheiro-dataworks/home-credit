"""Model evaluation utilities — builds a comprehensive report dict for the dashboard."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def build_eval_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Return a JSON-serialisable evaluation dict including curve data for charts."""
    y_pred = (y_prob >= threshold).astype(int)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    auc_roc = round(float(roc_auc_score(y_true, y_prob)), 4)
    auc_pr  = round(float(average_precision_score(y_true, y_prob)), 4)
    prec    = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    rec     = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1      = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── ROC curve (sampled to 200 pts for JSON size) ──────────────────────────
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
    roc_data = {
        "fpr": [round(float(v), 4) for v in fpr[idx]],
        "tpr": [round(float(v), 4) for v in tpr[idx]],
    }

    # ── Precision-Recall curve ────────────────────────────────────────────────
    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, y_prob)
    idx = np.linspace(0, len(pr_prec) - 1, min(200, len(pr_prec))).astype(int)
    pr_data = {
        "precision": [round(float(v), 4) for v in pr_prec[idx]],
        "recall":    [round(float(v), 4) for v in pr_rec[idx]],
    }

    # ── Calibration curve ─────────────────────────────────────────────────────
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    calib_data = {
        "mean_predicted": [round(float(v), 4) for v in mean_pred],
        "fraction_pos":   [round(float(v), 4) for v in frac_pos],
    }

    # ── Score distribution (histogram) ───────────────────────────────────────
    hist0, edges = np.histogram(y_prob[y_true == 0], bins=30, range=(0, 1))
    hist1, _     = np.histogram(y_prob[y_true == 1], bins=30, range=(0, 1))
    bin_centers  = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    score_dist   = {
        "bins":     [round(float(v), 4) for v in bin_centers],
        "non_default": [int(v) for v in hist0],
        "default":     [int(v) for v in hist1],
    }

    return {
        "auc_roc":            auc_roc,
        "auc_pr":             auc_pr,
        "precision":          prec,
        "recall":             rec,
        "f1":                 f1,
        "threshold":          round(float(threshold), 4),
        "confusion_matrix":   {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve":          roc_data,
        "pr_curve":           pr_data,
        "calibration_curve":  calib_data,
        "score_distribution": score_dist,
    }
