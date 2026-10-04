"""
Three-way, label-stratified train / calibration / test split.

Used once in the features stage, before any fitting step, so that imputation
medians, categorical encoders, and feature-selection statistics are derived
only from the training partition. See the README's "Validation strategy"
section for the rationale.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def three_way_split(
    index: pd.Index,
    target: pd.Series,
    calibration_size: float,
    test_size: float,
    random_state: int,
) -> dict[str, pd.Index]:
    """Split `index` into disjoint train/calibration/test index arrays,
    stratified on `target` at both splitting steps.

    Returns a dict with keys "train", "calibration", "test".
    """
    holdout_size = calibration_size + test_size
    if not 0 < holdout_size < 1:
        raise ValueError("calibration_size + test_size must be between 0 and 1")

    train_idx, holdout_idx = train_test_split(
        index, test_size=holdout_size, stratify=target, random_state=random_state
    )
    calib_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=calibration_size / holdout_size,
        stratify=target.loc[holdout_idx],
        random_state=random_state,
    )
    return {"train": train_idx, "calibration": calib_idx, "test": test_idx}
