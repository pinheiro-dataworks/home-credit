"""Guards against the two leakage modes fixed in the validation-strategy
rework: (1) the three-way split must be disjoint and exhaustive, and (2) a
transform fitted on train must be applied unchanged to calibration/test, never
re-fitted on them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import three_way_split
from src.features.engineering import encode_and_impute


@pytest.fixture
def synthetic_index_and_target():
    rng = np.random.default_rng(0)
    n = 1000
    target = pd.Series(rng.choice([0, 1], n, p=[0.9, 0.1]), index=pd.RangeIndex(n))
    return pd.RangeIndex(n), target


def test_three_way_split_is_disjoint_and_exhaustive(synthetic_index_and_target):
    index, target = synthetic_index_and_target
    splits = three_way_split(index, target, calibration_size=0.15, test_size=0.15, random_state=42)

    train_set = set(splits["train"])
    calib_set = set(splits["calibration"])
    test_set  = set(splits["test"])

    assert train_set.isdisjoint(calib_set)
    assert train_set.isdisjoint(test_set)
    assert calib_set.isdisjoint(test_set)
    assert train_set | calib_set | test_set == set(index)


def test_three_way_split_proportions_match_config(synthetic_index_and_target):
    index, target = synthetic_index_and_target
    splits = three_way_split(index, target, calibration_size=0.15, test_size=0.15, random_state=42)
    n = len(index)

    assert abs(len(splits["calibration"]) / n - 0.15) < 0.02
    assert abs(len(splits["test"]) / n - 0.15) < 0.02
    assert abs(len(splits["train"]) / n - 0.70) < 0.02


def test_three_way_split_is_stratified(synthetic_index_and_target):
    index, target = synthetic_index_and_target
    splits = three_way_split(index, target, calibration_size=0.15, test_size=0.15, random_state=42)
    overall_rate = target.mean()

    for name in ("train", "calibration", "test"):
        rate = target.loc[splits[name]].mean()
        assert abs(rate - overall_rate) < 0.05, f"{name} partition default rate diverges from overall"


def test_three_way_split_reproducible(synthetic_index_and_target):
    index, target = synthetic_index_and_target
    a = three_way_split(index, target, calibration_size=0.15, test_size=0.15, random_state=42)
    b = three_way_split(index, target, calibration_size=0.15, test_size=0.15, random_state=42)

    assert list(a["train"]) == list(b["train"])
    assert list(a["test"]) == list(b["test"])


def test_imputation_median_comes_from_train_not_from_transformed_partition():
    """The core leakage check: a NaN in the calibration partition must be filled
    with the TRAIN median, not a median derived from the calibration partition
    itself — even when the two partitions have very different distributions."""
    train_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100.0, 100.0, 100.0, 100.0, 100.0],   # train median = 100
        "CATEGORY":         ["A", "A", "B", "B", "A"],
    })
    calib_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [np.nan, np.nan, 9000.0],              # calib-only median would be 9000
        "CATEGORY":         ["A", "B", "A"],
    })

    _, state = encode_and_impute(train_df, fit=True)
    calib_encoded, _ = encode_and_impute(calib_df, fit=False, _state=state)

    assert (calib_encoded.loc[calib_df["AMT_INCOME_TOTAL"].isna(), "AMT_INCOME_TOTAL"] == 100.0).all()


def test_encode_and_impute_transform_does_not_mutate_fitted_state():
    train_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100.0, 200.0, 300.0],
        "CATEGORY":         ["A", "B", "A"],
    })
    calib_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [150.0],
        "CATEGORY":         ["__NEVER_SEEN__"],
    })

    _, state = encode_and_impute(train_df, fit=True)
    classes_before = list(state["label_encoders"]["CATEGORY"].classes_)
    medians_before = state["medians"].copy()

    encode_and_impute(calib_df, fit=False, _state=state)

    assert list(state["label_encoders"]["CATEGORY"].classes_) == classes_before
    assert state["medians"].equals(medians_before)
