"""
Model training pipeline:
  1. SMOTE / ADASYN resampling
  2. LightGBM with Optuna hyperparameter search + MedianPruner
  3. SHAP feature importance
  4. Probability calibration (isotonic regression)
  5. Threshold optimisation
  6. MLflow experiment tracking
"""
from __future__ import annotations
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import mlflow
import mlflow.lightgbm
import optuna
from optuna.pruners import MedianPruner
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from ..config import MODELS_DIR, PARAMS, RANDOM_STATE, EXPERIMENT_NAME, MLFLOW_URI
from ..utils.stats import (
    bootstrap_metrics, model_ks_statistic, find_optimal_threshold, build_statistical_report
)
from .evaluate import build_eval_report

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Resampling
# ──────────────────────────────────────────────────────────────────────────────

def resample(X: np.ndarray, y: np.ndarray, method: str, **kwargs) -> tuple:
    if method == "smote":
        sampler = SMOTE(
            sampling_strategy=PARAMS["sampling"]["sampling_strategy"],
            k_neighbors=PARAMS["sampling"]["k_neighbors"],
            random_state=RANDOM_STATE,
        )
    elif method == "adasyn":
        sampler = ADASYN(
            sampling_strategy=PARAMS["sampling"]["sampling_strategy"],
            n_neighbors=PARAMS["sampling"]["k_neighbors"],
            random_state=RANDOM_STATE,
        )
    else:
        return X, y

    X_res, y_res = sampler.fit_resample(X, y)
    logger.info("Resampling [%s]: %s → %s", method, dict(zip(*np.unique(y, return_counts=True))),
                dict(zip(*np.unique(y_res, return_counts=True))))
    return X_res, y_res


# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────────────

def _make_objective(X_train, y_train, cv: int, n_jobs: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state":      RANDOM_STATE,
            "n_jobs":            n_jobs,
            "verbose":           -1,
        }
        model = LGBMClassifier(**params)
        skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=skf,
                                  scoring="roc_auc", n_jobs=1)
        trial.report(scores.mean(), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return scores.mean()
    return objective


# ──────────────────────────────────────────────────────────────────────────────
# SHAP importance
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_importance(model, X_sample: pd.DataFrame, n_samples: int = 3000) -> list[dict]:
    sample = X_sample.sample(min(n_samples, len(X_sample)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    # LightGBM binary: sv is list of 2 arrays; take class-1 values
    if isinstance(sv, list):
        sv = sv[1]
    importance = np.abs(sv).mean(axis=0)
    feat_imp = [
        {"feature": col, "shap_importance": round(float(v), 6)}
        for col, v in zip(X_sample.columns, importance)
    ]
    feat_imp.sort(key=lambda x: x["shap_importance"], reverse=True)
    return feat_imp


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
) -> dict:
    """
    Full training run: Optuna → best LightGBM → calibration → SHAP → MLflow.
    Returns artefact dict with paths to saved files.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(MLFLOW_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    cfg       = PARAMS["model"]
    n_trials  = cfg["n_trials"]
    cv        = cfg["cv_folds"]
    n_jobs    = cfg["n_jobs"]
    method    = PARAMS["sampling"]["method"]
    thr_metric= PARAMS["threshold"]["metric"]

    with mlflow.start_run(run_name="lgbm_optuna") as run:
        # ── 1. Resample ──────────────────────────────────────────────────────
        X_res, y_res = resample(X_train.values, y_train.values, method)

        # ── 2. Optuna hyperparameter search ──────────────────────────────────
        pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        study   = optuna.create_study(
            direction="maximize",
            pruner=pruner if cfg["pruning"] else optuna.pruners.NopPruner(),
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            _make_objective(X_res, y_res, cv, n_jobs),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_params.update({"random_state": RANDOM_STATE, "n_jobs": n_jobs, "verbose": -1})
        mlflow.log_params(best_params)
        mlflow.log_param("sampling_method", method)
        mlflow.log_param("n_optuna_trials", n_trials)
        logger.info("Best AUC (CV): %.4f | params: %s", study.best_value, best_params)

        # ── 3. Final model fit ────────────────────────────────────────────────
        model = LGBMClassifier(**best_params)
        model.fit(
            X_res, y_res,
            eval_set=[(X_val.values, y_val.values)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )

        # ── 4. Probability calibration ────────────────────────────────────────
        calib_method = PARAMS["calibration"]["method"]
        calibrated   = CalibratedClassifierCV(model, method=calib_method, cv="prefit")
        calibrated.fit(X_val.values, y_val.values)

        # ── 5. Threshold optimisation ─────────────────────────────────────────
        y_prob_val = calibrated.predict_proba(X_val.values)[:, 1]
        opt_thr, opt_metric = find_optimal_threshold(y_val.values, y_prob_val, thr_metric)

        # ── 6. Evaluation & bootstrap CI ─────────────────────────────────────
        eval_report = build_eval_report(y_val.values, y_prob_val, opt_thr)
        ci_report   = bootstrap_metrics(
            y_val.values, y_prob_val, threshold=opt_thr,
            n_iterations=PARAMS["statistics"]["bootstrap_n_iterations"],
            ci=PARAMS["statistics"]["bootstrap_ci"],
        )
        ks_stat = model_ks_statistic(y_val.values, y_prob_val)

        # ── 7. SHAP importance ────────────────────────────────────────────────
        shap_imp = compute_shap_importance(model, X_val)

        # ── 8. Log metrics to MLflow ──────────────────────────────────────────
        flat_metrics = {
            "auc_roc":         eval_report["auc_roc"],
            "auc_pr":          eval_report["auc_pr"],
            "precision":       eval_report["precision"],
            "recall":          eval_report["recall"],
            "f1":              eval_report["f1"],
            "ks_statistic":    ks_stat,
            "optimal_threshold": opt_thr,
        }
        mlflow.log_metrics(flat_metrics)
        mlflow.lightgbm.log_model(model, "lgbm_model")

        # ── 9. Save artefacts ─────────────────────────────────────────────────
        joblib.dump(model,      MODELS_DIR / "model.pkl")
        joblib.dump(calibrated, MODELS_DIR / "calibrated_model.pkl")

        threshold_info = {"threshold": opt_thr, "metric": thr_metric, "value": opt_metric}
        (MODELS_DIR / "threshold.json").write_text(json.dumps(threshold_info, indent=2))
        (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names))
        (MODELS_DIR / "metrics.json").write_text(json.dumps(flat_metrics, indent=2))

        # Pre-computed stats for the dashboard API
        _save_precomputed_stats(
            eval_report, ci_report, shap_imp, ks_stat, opt_thr, run.info.run_id
        )

        logger.info("Training complete. Run ID: %s", run.info.run_id)
        return {
            "run_id": run.info.run_id,
            "metrics": flat_metrics,
            "shap_top10": shap_imp[:10],
        }


def _save_precomputed_stats(eval_report, ci_report, shap_imp, ks_stat, threshold, run_id):
    stats = {
        "run_id":    run_id,
        "threshold": threshold,
        "metrics":   eval_report,
        "bootstrap_ci": ci_report,
        "ks_statistic": ks_stat,
        "shap_importance": shap_imp[:50],
        "roc_curve":  eval_report.get("roc_curve"),
        "pr_curve":   eval_report.get("pr_curve"),
        "calibration_curve": eval_report.get("calibration_curve"),
        "confusion_matrix":  eval_report.get("confusion_matrix"),
        "score_distribution": eval_report.get("score_distribution"),
    }
    (MODELS_DIR / "precomputed_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Precomputed stats saved → models/artifacts/precomputed_stats.json")
