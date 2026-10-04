"""
DVC-compatible pipeline entry point.
Usage:
    python pipeline.py --stage features
    python pipeline.py --stage stats
    python pipeline.py --stage train
    python pipeline.py --stage predict
    python pipeline.py --stage drift
    python pipeline.py --stage all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import joblib

from src.config import (
    CALIBRATION_SIZE,
    DATA_FEATURES,
    DATA_PROCESSED,
    ID_COL,
    MODELS_DIR,
    NAN_THRESHOLD,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from src.data.loader import load_all
from src.data.splitting import three_way_split
from src.features.engineering import build_features, encode_and_impute
from src.models.drift import build_drift_report
from src.models.fairness import build_error_analysis, build_fairness_report
from src.models.train import train
from src.utils.stats import build_statistical_report

# Raw demographic/categorical columns used to slice fairness & error-analysis
# segments. DAYS_BIRTH is binned into age_band below rather than used raw.
FAIRNESS_SEGMENT_COLS = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"]
AGE_BAND_EDGES  = [0, 25, 35, 45, 55, 65, 200]
AGE_BAND_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# Features checked for drift (top model inputs + the sensitive attributes above)
DRIFT_FEATURE_CANDIDATES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
]

SPLIT_NAMES = ("train", "calibration", "test")


def _build_fairness_segments(df_raw_slice: pd.DataFrame, enc_state: dict) -> dict:
    """Decode label-encoded sensitive columns back to raw categories and derive
    an age band from DAYS_BIRTH, for use as fairness/error-analysis segments."""
    segments = {}
    encoders = enc_state.get("label_encoders", {})
    for col in FAIRNESS_SEGMENT_COLS:
        if col not in df_raw_slice.columns:
            continue
        enc = encoders.get(col)
        if enc is None:
            continue
        codes = df_raw_slice[col].astype(int).values
        segments[col] = pd.Series(enc.inverse_transform(codes))

    if "DAYS_BIRTH" in df_raw_slice.columns:
        age_years = -df_raw_slice["DAYS_BIRTH"].values / 365.25
        segments["age_band"] = pd.Series(
            pd.cut(age_years, bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS).astype(str)
        )

    return segments


# ──────────────────────────────────────────────────────────────────────────────

def stage_features():
    """Engineer features once, then split BEFORE any fitting step (imputation
    medians, categorical encoders) so that the calibration and test partitions
    never influence a transform applied to the training partition."""
    logger.info("=" * 60)
    logger.info("STAGE: Feature Engineering")
    logger.info("=" * 60)

    raw = load_all()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    full_df = build_features(
        app       = raw["train"],
        bureau    = raw["bureau"],
        bureau_bal= raw["bureau_bal"],
        prev      = raw["prev"],
        pos       = raw["pos_cash"],
        cc        = raw["credit_card"],
        inst      = raw["installments"],
        nan_threshold=NAN_THRESHOLD,
    )
    submission_df = build_features(
        app       = raw["test"],
        bureau    = raw["bureau"],
        bureau_bal= raw["bureau_bal"],
        prev      = raw["prev"],
        pos       = raw["pos_cash"],
        cc        = raw["credit_card"],
        inst      = raw["installments"],
        nan_threshold=NAN_THRESHOLD,
    )

    feature_cols = [c for c in full_df.columns if c not in [TARGET_COL, ID_COL]]
    submission_df = submission_df.reindex(columns=[ID_COL] + feature_cols)

    # ── Three-way, label-stratified split (unencoded, unimputed) ───────────────
    split_idx = three_way_split(
        full_df.index, full_df[TARGET_COL],
        calibration_size=CALIBRATION_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    train_idx = split_idx["train"]

    # ── Fit imputation/encoding on the training partition only ─────────────────
    X_train_raw, enc_state = encode_and_impute(full_df.loc[train_idx, feature_cols], fit=True)
    split_encoded = {"train": X_train_raw}
    for name in ("calibration", "test"):
        X_t, _ = encode_and_impute(full_df.loc[split_idx[name], feature_cols], fit=False, _state=enc_state)
        split_encoded[name] = X_t

    X_submission, _ = encode_and_impute(submission_df[feature_cols], fit=False, _state=enc_state)

    split_shapes = {}
    for name in SPLIT_NAMES:
        idx = split_idx[name]
        part = pd.concat([
            full_df.loc[idx, [ID_COL, TARGET_COL]].reset_index(drop=True),
            split_encoded[name].reset_index(drop=True),
        ], axis=1)
        part.to_parquet(DATA_PROCESSED / f"{name}_features.parquet", index=False)
        split_shapes[name] = {
            "shape": list(part.shape),
            "default_rate": round(float(part[TARGET_COL].mean()), 4),
        }
        logger.info("%-11s → %s (default rate %.4f)", name, part.shape, split_shapes[name]["default_rate"])

    submission_out = pd.concat([
        submission_df[[ID_COL]].reset_index(drop=True),
        X_submission.reset_index(drop=True),
    ], axis=1)
    submission_out.to_parquet(DATA_PROCESSED / "submission_features.parquet", index=False)
    joblib.dump(enc_state, MODELS_DIR / "enc_state.pkl")

    feat_meta = {
        "features": feature_cols,
        "n_features": len(feature_cols),
        "splits": split_shapes,
        "submission_shape": list(submission_out.shape),
    }
    (DATA_FEATURES / "feature_names.json").write_text(json.dumps(feat_meta, indent=2))
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_cols))

    logger.info("Features saved → %s", DATA_PROCESSED)


def stage_stats():
    """KS / Chi-square feature-selection report, computed on the training
    partition only so that feature-selection decisions never see calibration
    or test data."""
    logger.info("=" * 60)
    logger.info("STAGE: Statistical Analysis")
    logger.info("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    report = build_statistical_report(df, target=TARGET_COL)

    report["dataset_summary"] = {
        "n_train": int(len(df)),
        "n_features": int(df.shape[1] - 2),   # minus ID and TARGET
        "default_rate": round(float(df[TARGET_COL].mean()), 4),
        "default_count": int(df[TARGET_COL].sum()),
        "non_default_count": int((df[TARGET_COL] == 0).sum()),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "statistical_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Statistical report saved → models/artifacts/statistical_report.json")


def stage_train():
    logger.info("=" * 60)
    logger.info("STAGE: Model Training")
    logger.info("=" * 60)

    feats = json.loads((MODELS_DIR / "feature_names.json").read_text())

    splits = {}
    for name in SPLIT_NAMES:
        df = pd.read_parquet(DATA_PROCESSED / f"{name}_features.parquet")
        cols = [f for f in feats if f in df.columns]
        splits[name] = (df[cols], df[TARGET_COL])
    feats = [f for f in feats if f in splits["train"][0].columns]

    X_train, y_train = splits["train"]
    X_calib, y_calib = splits["calibration"]
    X_test,  y_test  = splits["test"]
    logger.info(
        "Train: %s | Calibration: %s | Test: %s | Positive rate (train): %.2f%%",
        X_train.shape, X_calib.shape, X_test.shape, y_train.mean() * 100,
    )

    result = train(X_train, y_train, X_calib, y_calib, X_test, y_test, feats)
    logger.info("Metrics (test partition): %s", result["metrics"])

    # ── Fairness / calibration-by-segment / FN-FP analysis ───────────────────
    # Computed on the test partition — the same one the headline metrics come
    # from — never on data used for fitting, early stopping or calibration.
    opt_thr = json.loads((MODELS_DIR / "threshold.json").read_text())["threshold"]
    y_prob_test = result["y_prob_test"]

    enc_state = joblib.load(MODELS_DIR / "enc_state.pkl") if (MODELS_DIR / "enc_state.pkl").exists() else {}
    segments  = _build_fairness_segments(X_test, enc_state)

    fairness_report = build_fairness_report(y_test.values, y_prob_test, opt_thr, segments) if segments else {}
    error_analysis  = build_error_analysis(y_test.values, y_prob_test, opt_thr, segments)

    stats_path = MODELS_DIR / "precomputed_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        n_total = len(X_train) + len(X_calib) + len(X_test)
        total_defaults = int(y_train.sum() + y_calib.sum() + y_test.sum())
        stats["dataset"] = {
            # Whole labeled population (train + calibration + test combined) —
            # the dashboard Overview page's portfolio-level KPIs. Since the
            # split is label-stratified, each partition's own default rate is
            # within noise of this value; this is the honest "how much data
            # did we have" figure rather than any one partition's slice of it.
            "n_train":           int(n_total),
            "n_features":        len(feats),
            "default_rate":      round(total_defaults / n_total, 4),
            "default_count":     total_defaults,
            "non_default_count": int(n_total - total_defaults),
            # Per-partition breakdown, for anything that specifically needs it.
            "splits": {
                "train":       int(len(X_train)),
                "calibration": int(len(X_calib)),
                "test":        int(len(X_test)),
            },
        }
        stats["fairness"]       = fairness_report
        stats["error_analysis"] = error_analysis

        if (MODELS_DIR / "statistical_report.json").exists():
            stats["statistical_report"] = json.loads((MODELS_DIR / "statistical_report.json").read_text())

        stats_path.write_text(json.dumps(stats, indent=2))
        logger.info("Fairness segments: %s | FN=%d FP=%d",
                    list(fairness_report.keys()),
                    error_analysis["overall"]["fn_count"], error_analysis["overall"]["fp_count"])


def stage_predict():
    """Score the Kaggle application_test holdout and write a submission file."""
    logger.info("=" * 60)
    logger.info("STAGE: Generate Submission")
    logger.info("=" * 60)

    sub_df = pd.read_parquet(DATA_PROCESSED / "submission_features.parquet")
    feats  = json.loads((MODELS_DIR / "feature_names.json").read_text())
    model  = joblib.load(MODELS_DIR / "calibrated_model.pkl")
    feats  = [f for f in feats if f in sub_df.columns]

    probs = model.predict_proba(sub_df[feats].values)[:, 1]
    sub   = pd.DataFrame({"SK_ID_CURR": sub_df[ID_COL], "TARGET": probs})
    (DATA_PROCESSED / "submission.csv").parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(DATA_PROCESSED / "submission.csv", index=False)
    logger.info("Submission saved: %d rows", len(sub))


def stage_drift():
    """Population Stability Index: training partition vs. the Kaggle
    application_test holdout, used as a covariate-shift proxy."""
    logger.info("=" * 60)
    logger.info("STAGE: Drift Monitoring (PSI)")
    logger.info("=" * 60)

    ref_df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    cur_df = pd.read_parquet(DATA_PROCESSED / "submission_features.parquet")
    features = [f for f in DRIFT_FEATURE_CANDIDATES if f in ref_df.columns and f in cur_df.columns]

    drift_report = build_drift_report(ref_df, cur_df, features)
    logger.info("Drift: %d checked | %d significant | %d moderate",
                drift_report["summary"]["n_features_checked"],
                drift_report["summary"]["n_significant"],
                drift_report["summary"]["n_moderate"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "drift_report.json").write_text(json.dumps(drift_report, indent=2))
    logger.info("Drift report saved → models/artifacts/drift_report.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features","stats","train","predict","drift","all"],
                        default="all")
    args = parser.parse_args()

    stages = {
        "features": stage_features,
        "stats":    stage_stats,
        "train":    stage_train,
        "predict":  stage_predict,
        "drift":    stage_drift,
    }

    if args.stage == "all":
        for fn in stages.values():
            fn()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
