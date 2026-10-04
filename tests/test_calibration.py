"""Probability calibration contract: output stays in [0, 1] and isotonic
calibration does not grotesquely worsen the Brier score on held-out data."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split


def _synthetic_split():
    X, y = make_classification(
        n_samples=3000, n_features=15, n_informative=8,
        weights=[0.88, 0.12], class_sep=1.2, random_state=42,
    )
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42
    )
    return X_train, y_train, X_calib, y_calib, X_test, y_test


def test_calibrated_probabilities_are_in_unit_interval():
    X_train, y_train, X_calib, y_calib, X_test, y_test = _synthetic_split()

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_calib, y_calib)

    probs = calibrated.predict_proba(X_test)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_calibration_does_not_grossly_worsen_brier_score():
    X_train, y_train, X_calib, y_calib, X_test, y_test = _synthetic_split()

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    uncalibrated_probs = model.predict_proba(X_test)[:, 1]
    uncalibrated_brier = brier_score_loss(y_test, uncalibrated_probs)

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_calib, y_calib)
    calibrated_probs = calibrated.predict_proba(X_test)[:, 1]
    calibrated_brier = brier_score_loss(y_test, calibrated_probs)

    # Isotonic calibration on a calibration fold can be marginally noisier than
    # the raw model on a well-separated synthetic dataset, but it must stay in
    # the same ballpark rather than degrading sharply.
    assert calibrated_brier < uncalibrated_brier * 1.5
