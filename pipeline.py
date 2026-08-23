"""
DVC-compatible pipeline entry point.
Usage:
    python pipeline.py --stage features
    python pipeline.py --stage stats
    python pipeline.py --stage train
    python pipeline.py --stage predict
    python pipeline.py --stage all
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_PROCESSED, DATA_FEATURES, MODELS_DIR,
    TARGET_COL, ID_COL, RANDOM_STATE, TEST_SIZE, NAN_THRESHOLD, PARAMS,
)
from src.data.loader import load_all
from src.features.engineering import build_features, encode_and_impute
from src.utils.stats import build_statistical_report
from src.models.train import train
from src.models.evaluate import build_eval_report
from src.models.fairness import build_fairness_report, build_error_analysis
from src.models.drift import build_drift_report
import joblib

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
    logger.info("=" * 60)
    logger.info("STAGE: Feature Engineering")
    logger.info("=" * 60)

    raw = load_all()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)

    train_df = build_features(
        app       = raw["train"],
        bureau    = raw["bureau"],
        bureau_bal= raw["bureau_bal"],
        prev      = raw["prev"],
        pos       = raw["pos_cash"],
        cc        = raw["credit_card"],
        inst      = raw["installments"],
        nan_threshold=NAN_THRESHOLD,
    )

    test_df = build_features(
        app       = raw["test"],
        bureau    = raw["bureau"],
        bureau_bal= raw["bureau_bal"],
        prev      = raw["prev"],
        pos       = raw["pos_cash"],
        cc        = raw["credit_card"],
        inst      = raw["installments"],
        nan_threshold=NAN_THRESHOLD,
    )

    # Align test columns to train (minus TARGET)
    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    test_df = test_df.reindex(columns=[ID_COL] + feature_cols)

    # Encode and impute
    X_train, enc_state = encode_and_impute(train_df[feature_cols], fit=True)
    X_test,  _         = encode_and_impute(test_df[feature_cols],  fit=False, _state=enc_state)

    train_out = pd.concat([
        train_df[[ID_COL, TARGET_COL]].reset_index(drop=True),
        X_train.reset_index(drop=True),
    ], axis=1)
    test_out  = pd.concat([
        test_df[[ID_COL]].reset_index(drop=True),
        X_test.reset_index(drop=True),
    ], axis=1)

    train_out.to_parquet(DATA_PROCESSED / "train_features.parquet", index=False)
    test_out.to_parquet (DATA_PROCESSED / "test_features.parquet",  index=False)
    joblib.dump(enc_state, MODELS_DIR / "enc_state.pkl")

    feat_meta = {
        "features": feature_cols,
        "n_features": len(feature_cols),
        "train_shape": list(train_out.shape),
        "test_shape":  list(test_out.shape),
        "target_distribution": {
            "0": round(float((train_df[TARGET_COL] == 0).mean()), 4),
            "1": round(float((train_df[TARGET_COL] == 1).mean()), 4),
        },
    }
    (DATA_FEATURES / "feature_names.json").write_text(json.dumps(feat_meta, indent=2))
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_cols))

    logger.info("Features saved → %s", DATA_PROCESSED)
    logger.info("Train: %s | Test: %s", train_out.shape, test_out.shape)


def stage_stats():
    logger.info("=" * 60)
    logger.info("STAGE: Statistical Analysis")
    logger.info("=" * 60)

    df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    report = build_statistical_report(df, target=TARGET_COL)

    # Append dataset summary
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

    df    = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    feats = json.loads((MODELS_DIR / "feature_names.json").read_text())

    # Use only columns present in df
    feats = [f for f in feats if f in df.columns]
    X     = df[feats]
    y     = df[TARGET_COL]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %s | Val: %s | Positive rate train: %.2f%%",
                X_tr.shape, X_val.shape, y_tr.mean() * 100)

    result = train(X_tr, y_tr, X_val, y_val, feats)
    logger.info("Metrics: %s", result["metrics"])

    # ── Fairness / calibration-by-segment / FN-FP analysis ───────────────────
    # Re-scored on the same validation split train() used, against the just-
    # saved calibrated model + optimal threshold — no retraining involved.
    calibrated = joblib.load(MODELS_DIR / "calibrated_model.pkl")
    opt_thr    = json.loads((MODELS_DIR / "threshold.json").read_text())["threshold"]
    y_prob_val = calibrated.predict_proba(X_val.values)[:, 1]

    enc_state = joblib.load(MODELS_DIR / "enc_state.pkl") if (MODELS_DIR / "enc_state.pkl").exists() else {}
    segments  = _build_fairness_segments(X_val, enc_state)

    fairness_report = build_fairness_report(y_val.values, y_prob_val, opt_thr, segments) if segments else {}
    error_analysis  = build_error_analysis(y_val.values, y_prob_val, opt_thr, segments)

    # Enrich precomputed_stats with dataset info
    stats_path = MODELS_DIR / "precomputed_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats["dataset"] = {
            "n_train":          int(len(df)),
            "n_val":            int(len(X_val)),
            "n_features":       len(feats),
            "default_rate":     round(float(y.mean()), 4),
            "default_count":    int(y.sum()),
            "non_default_count":int((y == 0).sum()),
        }
        stats["fairness"]       = fairness_report
        stats["error_analysis"] = error_analysis

        # Statistical report
        if (MODELS_DIR / "statistical_report.json").exists():
            stat_report = json.loads((MODELS_DIR / "statistical_report.json").read_text())
            stats["statistical_report"] = stat_report

        stats_path.write_text(json.dumps(stats, indent=2))
        logger.info("Fairness segments: %s | FN=%d FP=%d",
                    list(fairness_report.keys()),
                    error_analysis["overall"]["fn_count"], error_analysis["overall"]["fp_count"])


def stage_predict():
    logger.info("=" * 60)
    logger.info("STAGE: Generate Submission")
    logger.info("=" * 60)

    test_df = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    feats   = json.loads((MODELS_DIR / "feature_names.json").read_text())
    model   = joblib.load(MODELS_DIR / "calibrated_model.pkl")
    feats   = [f for f in feats if f in test_df.columns]

    probs   = model.predict_proba(test_df[feats].values)[:, 1]
    sub     = pd.DataFrame({"SK_ID_CURR": test_df[ID_COL], "TARGET": probs})
    (DATA_PROCESSED / "submission.csv").parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(DATA_PROCESSED / "submission.csv", index=False)
    logger.info("Submission saved: %d rows", len(sub))


def stage_drift():
    logger.info("=" * 60)
    logger.info("STAGE: Drift Monitoring (PSI)")
    logger.info("=" * 60)

    ref_df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    cur_df = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
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
