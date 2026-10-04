"""API contract tests via FastAPI's TestClient. Predictor falls back to mock
data when no trained model artefacts are present, so these tests exercise the
contract (status codes, response shape, value ranges) regardless of whether a
model has been trained in this environment."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

VALID_APPLICATION = {
    "AMT_INCOME_TOTAL": 135000.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 20560.0,
    "AMT_GOODS_PRICE": 351000.0,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "EXT_SOURCE_1": 0.502,
    "EXT_SOURCE_2": 0.626,
    "EXT_SOURCE_3": 0.555,
    "CNT_CHILDREN": 0,
    "CNT_FAM_MEMBERS": 2.0,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
}


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_predict_valid_payload_matches_schema():
    with TestClient(app) as client:
        resp = client.post("/api/predict", json=VALID_APPLICATION)
    assert resp.status_code == 200
    body = resp.json()

    assert 0.0 <= body["risk_score"] <= 100.0
    assert body["risk_label"] in {"Low", "Medium", "High"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert isinstance(body["predicted_default"], bool)
    assert 0.0 <= body["threshold"] <= 1.0


def test_predict_missing_required_field_is_rejected():
    payload = dict(VALID_APPLICATION)
    del payload["AMT_INCOME_TOTAL"]
    with TestClient(app) as client:
        resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_non_positive_income():
    payload = dict(VALID_APPLICATION)
    payload["AMT_INCOME_TOTAL"] = -1.0
    with TestClient(app) as client:
        resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422


def test_overview_schema():
    with TestClient(app) as client:
        resp = client.get("/api/overview")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("n_train", "n_test", "n_features", "default_rate", "default_count", "non_default_count"):
        assert key in body


def test_metrics_schema():
    with TestClient(app) as client:
        resp = client.get("/api/model/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["auc_roc"] <= 1.0
    assert 0.0 <= body["threshold"] <= 1.0
