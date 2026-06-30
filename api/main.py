"""
Home Credit Default Risk — FastAPI serving layer.
Deployed on Render free tier via Docker.
"""
from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    ApplicationInput, PredictionResponse,
    OverviewResponse, MetricsResponse, HealthResponse,
)
from .predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Home Credit Default Risk API",
    description=(
        "Open-finance credit risk prediction model. "
        "Trained on Home Credit Default Risk dataset with LightGBM + Optuna + SHAP."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton predictor loaded at startup
_predictor: Predictor | None = None


@app.on_event("startup")
async def startup():
    global _predictor
    _predictor = Predictor()
    logger.info("Predictor ready. Model loaded: %s", _predictor.is_loaded)


def _p() -> Predictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Service not ready.")
    return _predictor


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    return HealthResponse(status="ok", model_loaded=_p().is_loaded)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(status="ok", model_loaded=_p().is_loaded)


# ── Dataset overview ──────────────────────────────────────────────────────────

@app.get("/api/overview", response_model=OverviewResponse, tags=["Data"])
async def overview():
    return _p().get_overview()


# ── Model metrics ─────────────────────────────────────────────────────────────

@app.get("/api/model/metrics", response_model=MetricsResponse, tags=["Model"])
async def metrics():
    return _p().get_metrics()


@app.get("/api/model/roc", tags=["Model"])
async def roc_curve():
    return JSONResponse(_p().get_roc_curve())


@app.get("/api/model/pr-curve", tags=["Model"])
async def pr_curve():
    return JSONResponse(_p().get_pr_curve())


@app.get("/api/model/calibration", tags=["Model"])
async def calibration():
    return JSONResponse(_p().get_calibration_curve())


@app.get("/api/model/confusion-matrix", tags=["Model"])
async def confusion_matrix():
    return JSONResponse(_p().get_confusion_matrix())


# ── Feature importance ────────────────────────────────────────────────────────

@app.get("/api/features/importance", tags=["Features"])
async def feature_importance(top_n: int = 20):
    return JSONResponse({"importance": _p().get_shap_importance(top_n)})


# ── Risk distribution ─────────────────────────────────────────────────────────

@app.get("/api/risk/distribution", tags=["Risk"])
async def risk_distribution():
    return JSONResponse(_p().get_score_distribution())


# ── Statistical report ────────────────────────────────────────────────────────

@app.get("/api/statistics", tags=["Data"])
async def statistics():
    return JSONResponse(_p().get_statistical_report())


# ── Real-time prediction ──────────────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: ApplicationInput):
    result = _p().predict(application.model_dump())
    return PredictionResponse(**result)


# ── Sample applications (for dashboard table) ─────────────────────────────────

@app.get("/api/applications/sample", tags=["Data"])
async def sample_applications(n: int = 10):
    import random, math
    random.seed(42)
    apps = []
    for i in range(n):
        prob = round(random.uniform(0.02, 0.92), 4)
        apps.append({
            "id":          f"HC-{100_000 + i}",
            "income":      round(random.uniform(40_000, 300_000), 0),
            "credit":      round(random.uniform(80_000, 600_000), 0),
            "risk_score":  round(prob * 100, 1),
            "risk_label":  "Low" if prob < 0.25 else ("Medium" if prob < 0.55 else "High"),
            "predicted_default": prob >= 0.35,
            "contract_type": random.choice(["Cash loans", "Revolving loans"]),
            "submitted_at": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        })
    apps.sort(key=lambda x: x["risk_score"], reverse=True)
    return JSONResponse({"applications": apps, "total": n})
