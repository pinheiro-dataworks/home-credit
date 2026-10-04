"""Feature-engineering contract: shapes, no TARGET leakage into the feature
list, and train/submission column alignment."""
from __future__ import annotations

import numpy as np

from src.features.engineering import build_features, encode_and_impute


def _build(raw_tables, app_key):
    return build_features(
        app=raw_tables[app_key],
        bureau=raw_tables["bureau"],
        bureau_bal=raw_tables["bureau_bal"],
        prev=raw_tables["prev"],
        pos=raw_tables["pos_cash"],
        cc=raw_tables["credit_card"],
        inst=raw_tables["installments"],
        nan_threshold=0.6,
    )


def test_build_features_one_row_per_applicant(raw_tables):
    train_df = _build(raw_tables, "train")
    assert len(train_df) == len(raw_tables["train"])
    assert train_df["SK_ID_CURR"].is_unique


def test_target_present_only_when_input_has_it(raw_tables):
    train_df = _build(raw_tables, "train")
    test_df = _build(raw_tables, "test")
    assert "TARGET" in train_df.columns
    assert "TARGET" not in test_df.columns


def test_submission_columns_align_with_train_feature_columns(raw_tables):
    train_df = _build(raw_tables, "train")
    test_df = _build(raw_tables, "test")

    feature_cols = [c for c in train_df.columns if c not in ("TARGET", "SK_ID_CURR")]
    assert "TARGET" not in feature_cols

    aligned = test_df.reindex(columns=["SK_ID_CURR"] + feature_cols)
    assert list(aligned.columns) == ["SK_ID_CURR"] + feature_cols
    assert "TARGET" not in aligned.columns


def test_encode_and_impute_fit_produces_no_nan(raw_tables):
    train_df = _build(raw_tables, "train")
    feature_cols = [c for c in train_df.columns if c not in ("TARGET", "SK_ID_CURR")]

    X_encoded, state = encode_and_impute(train_df[feature_cols], fit=True)

    assert not X_encoded.isnull().any().any()
    assert set(X_encoded.select_dtypes(include=["object", "category"]).columns) == set()
    assert "label_encoders" in state and "medians" in state


def test_encode_and_impute_transform_handles_unseen_category(raw_tables):
    train_df = _build(raw_tables, "train")
    test_df  = _build(raw_tables, "test")
    feature_cols = [c for c in train_df.columns if c not in ("TARGET", "SK_ID_CURR")]
    test_df = test_df.reindex(columns=["SK_ID_CURR"] + feature_cols)

    _, state = encode_and_impute(train_df[feature_cols], fit=True)

    # Inject a category never seen during fit.
    test_slice = test_df[feature_cols].copy()
    test_slice.loc[test_slice.index[0], "NAME_EDUCATION_TYPE"] = "__UNSEEN_CATEGORY__"

    X_test, _ = encode_and_impute(test_slice, fit=False, _state=state)

    assert not X_test.isnull().any().any()
    assert np.issubdtype(X_test["NAME_EDUCATION_TYPE"].dtype, np.integer)
