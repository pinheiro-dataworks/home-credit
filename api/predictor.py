"""
Model loading and inference wrapper.
Falls back to mock data when model artefacts are absent (cold demo mode).
"""
from __future__ import annotations
import json
import logging
import math
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models" / "artifacts"


class Predictor:
    def __init__(self):
        self.model       = None
        self.features    = []
        self.threshold   = 0.35
        self.stats       = {}
        self.drift       = {}
        self._load()

    def _load(self):
        try:
            import joblib
            model_path  = MODELS_DIR / "calibrated_model.pkl"
            feat_path   = MODELS_DIR / "feature_names.json"
            thr_path    = MODELS_DIR / "threshold.json"
            stats_path  = MODELS_DIR / "precomputed_stats.json"
            drift_path  = MODELS_DIR / "drift_report.json"

            if model_path.exists():
                self.model     = joblib.load(model_path)
                self.features  = json.loads(feat_path.read_text()) if feat_path.exists() else []
                self.threshold = json.loads(thr_path.read_text())["threshold"] if thr_path.exists() else 0.35
                logger.info("Model loaded. Features: %d | Threshold: %.4f",
                            len(self.features), self.threshold)
            else:
                logger.warning("Model not found — serving mock data.")

            if stats_path.exists():
                self.stats = json.loads(stats_path.read_text())
            if drift_path.exists():
                self.drift = json.loads(drift_path.read_text())
        except Exception as exc:
            logger.error("Model load error: %s", exc)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, input_dict: dict) -> dict:
        if not self.is_loaded:
            return self._mock_prediction(input_dict)

        import pandas as pd
        row = pd.DataFrame([input_dict])

        # Engineer the same features as training
        from src.features.engineering import engineer_application
        row = engineer_application(row)

        # Align to training feature set
        missing = [f for f in self.features if f not in row.columns]
        for m in missing:
            row[m] = np.nan
        row = row[self.features]
        row = row.fillna(row.median())

        prob  = float(self.model.predict_proba(row.values)[0, 1])
        label = "Low" if prob < 0.25 else ("Medium" if prob < 0.55 else "High")

        return {
            "risk_score":          round(prob * 100, 2),
            "risk_label":          label,
            "default_probability": round(prob, 4),
            "threshold":           self.threshold,
            "predicted_default":   bool(prob >= self.threshold),
        }

    # ── Precomputed stats getters ─────────────────────────────────────────────

    def get_overview(self) -> dict:
        s = self.stats
        d = s.get("dataset", {})
        return {
            "n_train":               d.get("n_train", 307_511),
            "n_test":                48_744,
            "n_features":            d.get("n_features", 122),
            "default_rate":          d.get("default_rate", 0.0808),
            "default_count":         d.get("default_count", 24_825),
            "non_default_count":     d.get("non_default_count", 282_686),
            "n_engineered_features": len(self.features) if self.features else 200,
        }

    def get_metrics(self) -> dict:
        m   = self.stats.get("metrics", {})
        ci  = self.stats.get("bootstrap_ci", {})
        return {
            "auc_roc":      m.get("auc_roc",      0.781),
            "auc_pr":       m.get("auc_pr",        0.312),
            "precision":    m.get("precision",     0.453),
            "recall":       m.get("recall",        0.672),
            "f1":           m.get("f1",            0.541),
            "ks_statistic": self.stats.get("ks_statistic", 0.424),
            "threshold":    self.stats.get("threshold",     0.350),
            "bootstrap_ci": ci,
        }

    def get_shap_importance(self, top_n: int = 20) -> list[dict]:
        items = self.stats.get("shap_importance", _MOCK_SHAP)
        return items[:top_n]

    def get_roc_curve(self) -> dict:
        return self.stats.get("roc_curve", _MOCK_ROC)

    def get_pr_curve(self) -> dict:
        return self.stats.get("pr_curve", _MOCK_PR)

    def get_calibration_curve(self) -> dict:
        return self.stats.get("calibration_curve", _MOCK_CALIB)

    def get_score_distribution(self) -> dict:
        return self.stats.get("score_distribution", _MOCK_SCORE_DIST)

    def get_confusion_matrix(self) -> dict:
        return self.stats.get("confusion_matrix", {"tn": 52_100, "fp": 2_400, "fn": 1_900, "tp": 5_200})

    def get_statistical_report(self) -> dict:
        return self.stats.get("statistical_report", {"ks_tests": [], "chi2_tests": []})

    def get_fairness_report(self) -> dict:
        return self.stats.get("fairness") or _MOCK_FAIRNESS

    def get_error_analysis(self) -> dict:
        return self.stats.get("error_analysis") or _MOCK_ERROR_ANALYSIS

    def get_drift_report(self) -> dict:
        return self.drift or _MOCK_DRIFT

    # ── Mock data (demo mode when model hasn't been trained yet) ─────────────

    def _mock_prediction(self, inp: dict) -> dict:
        ext = inp.get("EXT_SOURCE_2", 0.5) or 0.5
        prob = round(max(0.01, min(0.99, 1 - ext + 0.05)), 4)
        label = "Low" if prob < 0.25 else ("Medium" if prob < 0.55 else "High")
        return {
            "risk_score":          round(prob * 100, 2),
            "risk_label":          label,
            "default_probability": prob,
            "threshold":           self.threshold,
            "predicted_default":   bool(prob >= self.threshold),
        }


# ── Fallback mock curves (guarantee non-empty charts) ─────────────────────────

def _roc_pts(n=40):
    t = np.linspace(0, 1, n)
    tpr = np.clip(t ** 0.4, 0, 1)
    return {"fpr": t.round(3).tolist(), "tpr": tpr.round(3).tolist()}

def _pr_pts(n=40):
    r = np.linspace(0, 1, n)
    p = np.clip(0.85 - 0.7 * r, 0.05, 1)
    return {"recall": r.round(3).tolist(), "precision": p.round(3).tolist()}

def _calib_pts():
    mp = [0.05,0.12,0.20,0.28,0.36,0.44,0.52,0.60,0.72,0.88]
    fp = [0.04,0.11,0.19,0.30,0.38,0.46,0.55,0.62,0.74,0.87]
    return {"mean_predicted": mp, "fraction_pos": fp}

def _score_dist():
    bins  = [round(i/30, 3) for i in range(30)]
    neg   = [int(v) for v in np.random.default_rng(0).integers(1000,8000,30)]
    pos   = [int(v) for v in np.random.default_rng(1).integers(50,800,30)]
    return {"bins": bins, "non_default": neg, "default": pos}

_MOCK_ROC        = _roc_pts()
_MOCK_PR         = _pr_pts()
_MOCK_CALIB      = _calib_pts()
_MOCK_SCORE_DIST = _score_dist()
_MOCK_SHAP = [
    {"feature": "EXT_SOURCE_2",        "shap_importance": 0.0842},
    {"feature": "EXT_SOURCE_3",        "shap_importance": 0.0721},
    {"feature": "EXT_SOURCE_1",        "shap_importance": 0.0614},
    {"feature": "DAYS_BIRTH",          "shap_importance": 0.0512},
    {"feature": "CREDIT_INCOME_RATIO", "shap_importance": 0.0487},
    {"feature": "ANNUITY_INCOME_RATIO","shap_importance": 0.0431},
    {"feature": "BUREAU_LOAN_COUNT",   "shap_importance": 0.0398},
    {"feature": "INSTAL_DPD_MEAN",     "shap_importance": 0.0376},
    {"feature": "DAYS_EMPLOYED",       "shap_importance": 0.0354},
    {"feature": "AMT_CREDIT",          "shap_importance": 0.0321},
    {"feature": "CREDIT_TERM",         "shap_importance": 0.0298},
    {"feature": "BUREAU_AMT_DEBT_MEAN","shap_importance": 0.0267},
    {"feature": "POS_DPD_MAX",         "shap_importance": 0.0243},
    {"feature": "CC_DPD_MEAN",         "shap_importance": 0.0221},
    {"feature": "PREV_REFUSED_RATIO",  "shap_importance": 0.0198},
    {"feature": "AMT_INCOME_TOTAL",    "shap_importance": 0.0187},
    {"feature": "EMPLOYED_TO_AGE_RATIO","shap_importance": 0.0165},
    {"feature": "BUREAU_OVERDUE_MAX",  "shap_importance": 0.0154},
    {"feature": "INSTAL_LATE_RATIO",   "shap_importance": 0.0143},
    {"feature": "EXT_SOURCE_MEAN",     "shap_importance": 0.0131},
]


# ── Fallback fairness / error-analysis / drift (illustrative, pre-training) ──

def _fairness_group(n, pos, neg, tp, fn, fp, tn, calib):
    return {
        "n": n,
        "base_rate":      round(pos / n, 4),
        "selection_rate": round((tp + fp) / n, 4),
        "tpr":            round(tp / pos, 4),
        "fpr":            round(fp / neg, 4),
        "precision":      round(tp / (tp + fp), 4),
        "auc_roc":        None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "calibration_curve": calib,
    }

_MOCK_FAIRNESS = {
    "CODE_GENDER": {
        "groups": {
            "F": _fairness_group(45_200, 3_571, 41_629, 2_432, 1_139, 6_328, 35_301,
                                  {"mean_predicted": [0.03,0.11,0.22,0.38,0.66],
                                   "fraction_pos":   [0.02,0.10,0.21,0.37,0.64]}),
            "M": _fairness_group(23_800, 2_118, 21_682, 1_408,   710, 3_708, 17_974,
                                  {"mean_predicted": [0.04,0.13,0.25,0.41,0.69],
                                   "fraction_pos":   [0.03,0.12,0.24,0.40,0.68]}),
        },
        "disparate_impact_ratio": 0.8934,
        "equal_opportunity_diff": 0.0155,
    },
    "age_band": {
        "groups": {
            "18-24": _fairness_group(3_100,   409, 2_691,   291,  118,   533, 2_158,
                                      {"mean_predicted": [0.06,0.18,0.32,0.50,0.78],
                                       "fraction_pos":   [0.09,0.24,0.40,0.55,0.74]}),
            "25-34": _fairness_group(18_200, 1_711, 16_489, 1_179,  532, 2_655, 13_834,
                                      {"mean_predicted": [0.04,0.14,0.26,0.42,0.70],
                                       "fraction_pos":   [0.04,0.13,0.25,0.41,0.69]}),
            "35-44": _fairness_group(16_400, 1_328, 15_072,   892,  436, 2_246, 12_826,
                                      {"mean_predicted": [0.03,0.12,0.23,0.39,0.67],
                                       "fraction_pos":   [0.03,0.11,0.22,0.38,0.66]}),
            "45-54": _fairness_group(13_100,   930, 12_170,   612,  318, 1_679, 10_491,
                                      {"mean_predicted": [0.03,0.10,0.20,0.35,0.63],
                                       "fraction_pos":   [0.02,0.09,0.19,0.34,0.62]}),
            "55-64": _fairness_group(8_900,    552,  8_348,   354,  198, 1_010,  7_338,
                                      {"mean_predicted": [0.02,0.09,0.18,0.32,0.60],
                                       "fraction_pos":   [0.02,0.08,0.17,0.31,0.58]}),
            "65+":   _fairness_group(2_100,    122,  1_978,    73,   49,   253,  1_725,
                                      {"mean_predicted": [0.02,0.08,0.17,0.30,0.58],
                                       "fraction_pos":   [0.04,0.13,0.24,0.38,0.62]}),
        },
        "disparate_impact_ratio": 0.4428,
        "equal_opportunity_diff": 0.1136,
    },
}

_MOCK_ERROR_ANALYSIS = {
    "overall": {
        "fn_count": 1_900, "fp_count": 2_400,
        "fn_rate": 0.268, "fp_rate": 0.044,
        "fn_avg_probability": 0.291, "fp_avg_probability": 0.432,
        "tn_avg_probability": 0.081, "tp_avg_probability": 0.628,
    },
    "by_segment": {
        "CODE_GENDER": {
            "F": {"n": 45_200, "fn_rate": 0.319, "fp_rate": 0.152},
            "M": {"n": 23_800, "fn_rate": 0.335, "fp_rate": 0.171},
        },
        "age_band": {
            "18-24": {"n": 3_100,  "fn_rate": 0.288, "fp_rate": 0.198},
            "25-34": {"n": 18_200, "fn_rate": 0.311, "fp_rate": 0.161},
            "35-44": {"n": 16_400, "fn_rate": 0.328, "fp_rate": 0.149},
            "45-54": {"n": 13_100, "fn_rate": 0.342, "fp_rate": 0.138},
            "55-64": {"n": 8_900,  "fn_rate": 0.359, "fp_rate": 0.121},
            "65+":   {"n": 2_100,  "fn_rate": 0.402, "fp_rate": 0.128},
        },
    },
}

_MOCK_DRIFT = {
    "reference_population": "application_train (training population)",
    "current_population": "application_test (Kaggle holdout, used as a covariate-shift proxy — no live production stream is connected yet)",
    "features": [
        {"feature": "DAYS_EMPLOYED",      "psi": 0.312, "status": "significant"},
        {"feature": "AMT_INCOME_TOTAL",   "psi": 0.187, "status": "moderate"},
        {"feature": "NAME_INCOME_TYPE",   "psi": 0.143, "status": "moderate"},
        {"feature": "AMT_CREDIT",         "psi": 0.081, "status": "stable"},
        {"feature": "AMT_ANNUITY",        "psi": 0.067, "status": "stable"},
        {"feature": "EXT_SOURCE_1",       "psi": 0.054, "status": "stable"},
        {"feature": "EXT_SOURCE_2",       "psi": 0.041, "status": "stable"},
        {"feature": "EXT_SOURCE_3",       "psi": 0.038, "status": "stable"},
        {"feature": "DAYS_BIRTH",         "psi": 0.029, "status": "stable"},
        {"feature": "CODE_GENDER",        "psi": 0.012, "status": "stable"},
        {"feature": "NAME_EDUCATION_TYPE","psi": 0.019, "status": "stable"},
        {"feature": "AMT_GOODS_PRICE",    "psi": 0.076, "status": "stable"},
    ],
    "summary": {"n_features_checked": 12, "n_significant": 1, "n_moderate": 2, "n_stable": 9},
}
