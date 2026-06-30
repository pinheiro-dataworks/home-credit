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
import joblib


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
        # Statistical report
        if (MODELS_DIR / "statistical_report.json").exists():
            stat_report = json.loads((MODELS_DIR / "statistical_report.json").read_text())
            stats["statistical_report"] = stat_report

        stats_path.write_text(json.dumps(stats, indent=2))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features","stats","train","predict","all"],
                        default="all")
    args = parser.parse_args()

    stages = {
        "features": stage_features,
        "stats":    stage_stats,
        "train":    stage_train,
        "predict":  stage_predict,
    }

    if args.stage == "all":
        for fn in stages.values():
            fn()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
