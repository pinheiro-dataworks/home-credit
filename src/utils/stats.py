"""
Statistical analysis module.
- Kolmogorov-Smirnov test (numeric features vs TARGET)
- Chi-square test (categorical features vs TARGET)
- Bootstrap confidence intervals for model metrics
- KS statistic for model discrimination (population stability)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Feature-level tests
# ──────────────────────────────────────────────────────────────────────────────

def ks_test_features(
    df: pd.DataFrame,
    target: str = "TARGET",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Two-sample KS test for each numeric column between TARGET=0 and TARGET=1.
    Returns DataFrame sorted by KS statistic (most discriminative first).
    """
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]

    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target and df[c].nunique() > 2
    ]

    records = []
    for col in num_cols:
        a = neg[col].dropna()
        b = pos[col].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        stat, pval = scipy_stats.ks_2samp(a, b)
        records.append({
            "feature": col,
            "ks_statistic": round(stat, 4),
            "p_value": round(pval, 6),
            "significant": pval < alpha,
            "mean_target0": round(a.mean(), 4),
            "mean_target1": round(b.mean(), 4),
        })

    result = (
        pd.DataFrame(records)
        .sort_values("ks_statistic", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("KS test: %d features tested, %d significant.",
                len(result), result["significant"].sum())
    return result


def chi2_test_features(
    df: pd.DataFrame,
    target: str = "TARGET",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Chi-square test of independence for each categorical column vs TARGET.
    Returns DataFrame sorted by chi2 statistic.
    """
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target
    ]
    # Also test low-cardinality numeric flags
    flag_cols = [
        c for c in df.select_dtypes(include="number").columns
        if df[c].nunique() <= 10 and c != target
    ]
    all_cats = list(set(cat_cols + flag_cols))

    records = []
    for col in all_cats:
        ct = pd.crosstab(df[col].fillna("MISSING"), df[target])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, pval, dof, _ = scipy_stats.chi2_contingency(ct)
        records.append({
            "feature": col,
            "chi2_statistic": round(chi2, 4),
            "p_value": round(pval, 6),
            "degrees_of_freedom": dof,
            "significant": pval < alpha,
        })

    result = (
        pd.DataFrame(records)
        .sort_values("chi2_statistic", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Chi2 test: %d features tested, %d significant.",
                len(result), result["significant"].sum())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals for model metrics
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_iterations: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals for AUC-ROC, AUC-PR, Precision, Recall, F1.
    Returns dict with point estimates and [lower, upper] CI for each metric.
    """
    rng = np.random.default_rng(random_state)
    y_pred = (y_prob >= threshold).astype(int)

    def _compute(yt, yp_prob, yp):
        return {
            "auc_roc":   roc_auc_score(yt, yp_prob),
            "auc_pr":    average_precision_score(yt, yp_prob),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall":    recall_score(yt, yp, zero_division=0),
            "f1":        f1_score(yt, yp, zero_division=0),
        }

    point = _compute(y_true, y_prob, y_pred)

    n = len(y_true)
    boot_results = {k: [] for k in point}
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true[idx]
        if yt_b.sum() == 0 or yt_b.sum() == n:
            continue
        yp_b = y_prob[idx]
        yp_b_pred = (yp_b >= threshold).astype(int)
        m = _compute(yt_b, yp_b, yp_b_pred)
        for k, v in m.items():
            boot_results[k].append(v)

    alpha = (1 - ci) / 2
    out = {}
    for k, vals in boot_results.items():
        arr = np.array(vals)
        out[k] = {
            "point":  round(point[k], 4),
            "lower":  round(float(np.quantile(arr, alpha)), 4),
            "upper":  round(float(np.quantile(arr, 1 - alpha)), 4),
            "ci":     ci,
        }
    logger.info("Bootstrap CI computed for %d iterations.", n_iterations)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model-level KS statistic (population stability)
# ──────────────────────────────────────────────────────────────────────────────

def model_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS statistic as used in credit scoring:
    max separation between CDF of P(score | bad) and CDF of P(score | good).
    """
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    ks_stat, _ = scipy_stats.ks_2samp(neg_scores, pos_scores)
    return round(float(ks_stat), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Optimal threshold via F1 / KS / Precision-Recall
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """
    Grid-search over [0.01, 0.99] for the threshold maximising `metric`.
    Returns (optimal_threshold, best_metric_value).
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_val, best_thr = -1.0, 0.5

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            val = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            val = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "ks":
            val = model_ks_statistic(y_true, y_prob)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if val > best_val:
            best_val, best_thr = val, thr

    logger.info("Optimal threshold (metric=%s): %.4f → value=%.4f", metric, best_thr, best_val)
    return round(float(best_thr), 4), round(float(best_val), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Full statistical report
# ──────────────────────────────────────────────────────────────────────────────

def build_statistical_report(
    df: pd.DataFrame,
    target: str = "TARGET",
    alpha: float = 0.05,
) -> dict:
    """Run all statistical tests and return a JSON-serialisable summary."""
    ks_df   = ks_test_features(df, target, alpha)
    chi2_df = chi2_test_features(df, target, alpha)

    return {
        "ks_tests": ks_df.head(30).to_dict(orient="records"),
        "chi2_tests": chi2_df.head(30).to_dict(orient="records"),
        "ks_tests_significant_count": int(ks_df["significant"].sum()),
        "chi2_tests_significant_count": int(chi2_df["significant"].sum()),
        "total_features_tested": len(ks_df) + len(chi2_df),
    }
