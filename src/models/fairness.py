"""
Subgroup fairness, per-segment calibration, and false positive / false negative
error analysis — evaluated on the held-out validation set.

Metrics follow standard fairness-in-ML definitions:
  - selection_rate         : share of a group flagged high-risk (demographic parity)
  - disparate_impact_ratio : min(selection_rate) / max(selection_rate) across groups
                             in a segment (four-fifths rule: flag if < 0.8)
  - tpr (recall)           : equal-opportunity metric — parity target is a small
                             `equal_opportunity_diff` (max TPR - min TPR)
  - fpr                    : false-positive-rate parity
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve

MIN_GROUP_SIZE = 30


def _group_report(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    tpr = round(tp / n_pos, 4) if n_pos else None
    fpr = round(fp / n_neg, 4) if n_neg else None
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    auc = round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else None

    calib = {"mean_predicted": [], "fraction_pos": []}
    if len(np.unique(y_true)) > 1 and len(y_true) >= 50:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="quantile")
        calib = {
            "mean_predicted": [round(float(v), 4) for v in mean_pred],
            "fraction_pos":   [round(float(v), 4) for v in frac_pos],
        }

    return {
        "n": len(y_true),
        "base_rate":       round(float(y_true.mean()), 4),
        "selection_rate":  round(float(y_pred.mean()), 4),
        "tpr":             tpr,
        "fpr":             fpr,
        "precision":       precision,
        "auc_roc":         auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration_curve": calib,
    }


def build_fairness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    segments: dict[str, pd.Series],
) -> dict:
    """
    segments: {segment_name: pd.Series of raw group labels}, index-aligned with
    y_true/y_prob (same order, same length). Groups smaller than MIN_GROUP_SIZE
    are dropped to avoid unstable estimates.
    """
    y_pred = (y_prob >= threshold).astype(int)
    report: dict = {}

    for seg_name, seg_values in segments.items():
        values = seg_values.reset_index(drop=True)
        groups: dict = {}
        for group in sorted(values.dropna().unique(), key=str):
            mask = (values == group).values
            if mask.sum() < MIN_GROUP_SIZE:
                continue
            groups[str(group)] = _group_report(y_true[mask], y_prob[mask], y_pred[mask])

        if len(groups) < 2:
            continue

        sel_rates = [g["selection_rate"] for g in groups.values()]
        tprs      = [g["tpr"] for g in groups.values() if g["tpr"] is not None]
        report[seg_name] = {
            "groups": groups,
            "disparate_impact_ratio": round(min(sel_rates) / max(sel_rates), 4) if max(sel_rates) > 0 else None,
            "equal_opportunity_diff": round(max(tprs) - min(tprs), 4) if len(tprs) >= 2 else None,
        }

    return report


def build_error_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    segments: dict[str, pd.Series] | None = None,
) -> dict:
    """Overall + per-segment false negative / false positive breakdown."""
    y_pred = (y_prob >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())

    def _avg_prob(mask):
        return round(float(y_prob[mask].mean()), 4) if mask.any() else None

    overall = {
        "fn_count": int(fn_mask.sum()),
        "fp_count": int(fp_mask.sum()),
        "fn_rate": round(fn_mask.sum() / n_pos, 4) if n_pos else None,
        "fp_rate": round(fp_mask.sum() / n_neg, 4) if n_neg else None,
        "fn_avg_probability": _avg_prob(fn_mask),
        "fp_avg_probability": _avg_prob(fp_mask),
        "tn_avg_probability": _avg_prob(tn_mask),
        "tp_avg_probability": _avg_prob(tp_mask),
    }

    by_segment: dict = {}
    if segments:
        for seg_name, seg_values in segments.items():
            values = seg_values.reset_index(drop=True)
            seg_report = {}
            for group in sorted(values.dropna().unique(), key=str):
                mask = (values == group).values
                if mask.sum() < MIN_GROUP_SIZE:
                    continue
                yt, yp = y_true[mask], y_pred[mask]
                fn_g = (yt == 1) & (yp == 0)
                fp_g = (yt == 0) & (yp == 1)
                gp, gn = int((yt == 1).sum()), int((yt == 0).sum())
                seg_report[str(group)] = {
                    "n": int(mask.sum()),
                    "fn_rate": round(fn_g.sum() / gp, 4) if gp else None,
                    "fp_rate": round(fp_g.sum() / gn, 4) if gn else None,
                }
            if seg_report:
                by_segment[seg_name] = seg_report

    return {"overall": overall, "by_segment": by_segment}
