"""
Data drift monitoring via Population Stability Index (PSI).

Reference population : application_train (the population the model was fit on).
Current population    : application_test (Kaggle's held-out set) is used as a
                         proxy for "newly arriving applications" — there is no
                         live production stream to compare against yet, so this
                         is a covariate-shift check, not a true production monitor.

Standard PSI thresholds (credit-risk industry convention):
  < 0.10  stable      — no material shift
  < 0.25  moderate     — investigate
  >= 0.25 significant  — distribution has shifted meaningfully, consider retraining
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STABLE_MAX   = 0.10
MODERATE_MAX = 0.25


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float | None:
    """PSI of `actual` vs `expected`, binned on `expected`'s quantiles."""
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual   = pd.to_numeric(actual,   errors="coerce").dropna()
    if len(expected) < bins or len(actual) == 0:
        return None

    quantiles   = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual,   bins=breakpoints)
    e_pct = np.clip(e_counts / e_counts.sum(), 1e-4, None)
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-4, None)

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return round(psi, 4)


def psi_status(psi: float | None) -> str:
    if psi is None:
        return "unknown"
    if psi < STABLE_MAX:
        return "stable"
    if psi < MODERATE_MAX:
        return "moderate"
    return "significant"


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
) -> dict:
    rows = []
    for f in features:
        if f not in reference_df.columns or f not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[f], current_df[f])
        rows.append({"feature": f, "psi": psi, "status": psi_status(psi)})

    rows.sort(key=lambda r: (r["psi"] is None, -(r["psi"] or 0)))
    n_significant = sum(1 for r in rows if r["status"] == "significant")
    n_moderate    = sum(1 for r in rows if r["status"] == "moderate")

    return {
        "reference_population": "application_train (training population)",
        "current_population":   "application_test (Kaggle holdout, used as a covariate-shift proxy — no live production stream is connected yet)",
        "features": rows,
        "summary": {
            "n_features_checked": len(rows),
            "n_significant": n_significant,
            "n_moderate":    n_moderate,
            "n_stable":      len(rows) - n_significant - n_moderate,
        },
    }
