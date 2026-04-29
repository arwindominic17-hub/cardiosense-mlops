"""
app.py
======
FastAPI REST API for CardioSense AI model serving.
Exposes /predict, /predict/batch, /health, /metrics, and /model-info endpoints.
Auto-generated docs available at /docs and /redoc.
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Load production bundle ──────────────────────────────────────
BUNDLE_PATH = os.environ.get("BUNDLE_PATH", "models/production_bundle.pkl")

MODEL    = None
SCALER   = None
FEATURES = None
THRESHOLD = 0.5

try:
    bundle   = joblib.load(BUNDLE_PATH)
    MODEL    = bundle["model"]
    SCALER   = bundle["scaler"]
    FEATURES = bundle["features"]
    THRESHOLD = float(bundle.get("threshold", 0.5))
    log.info(f"Model loaded from: {BUNDLE_PATH}")
    log.info(f"Model type: {type(MODEL).__name__}")
    log.info(f"Decision threshold: {THRESHOLD}")
except Exception as e:
    log.error(f"Failed to load model bundle: {e}")

# ── Runtime stats ───────────────────────────────────────────────
STATS = {
    "total_requests":   0,
    "successful_preds": 0,
    "failed_preds":     0,
    "high_risk_count":  0,
    "low_risk_count":   0,
    "start_time":       datetime.now(timezone.utc).isoformat(),
}

FEATURE_RANGES = {
    "age":      (20, 100),  "sex":      (0, 1),
    "cp":       (0, 3),     "trestbps": (80, 220),
    "chol":     (100, 600), "fbs":      (0, 1),
    "restecg":  (0, 2),     "thalach":  (60, 210),
    "exang":    (0, 1),     "oldpeak":  (0.0, 7.0),
    "slope":    (0, 2),     "ca":       (0, 4),
    "thal":     (0, 2),
}

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


# ── Pydantic Models ─────────────────────────────────────────────

class PatientInput(BaseModel):
    age:      float = Field(..., ge=20,  le=100,  description="Age in years")
    sex:      float = Field(..., ge=0,   le=1,    description="Sex (0=female, 1=male)")
    cp:       float = Field(..., ge=0,   le=3,    description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=80,  le=220,  description="Resting blood pressure (mmHg)")
    chol:     float = Field(..., ge=100, le=600,  description="Serum cholesterol (mg/dl)")
    fbs:      float = Field(..., ge=0,   le=1,    description="Fasting blood sugar > 120 mg/dl")
    restecg:  float = Field(..., ge=0,   le=2,    description="Resting ECG results (0-2)")
    thalach:  float = Field(..., ge=60,  le=210,  description="Max heart rate achieved")
    exang:    float = Field(..., ge=0,   le=1,    description="Exercise induced angina (0/1)")
    oldpeak:  float = Field(..., ge=0.0, le=7.0,  description="ST depression induced by exercise")
    slope:    float = Field(..., ge=0,   le=2,    description="Slope of peak exercise ST segment")
    ca:       float = Field(..., ge=0,   le=4,    description="Number of major vessels (0-4)")
    thal:     float = Field(..., ge=0,   le=2,    description="Thalassemia (0-2)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 58, "sex": 1, "cp": 0, "trestbps": 140,
                "chol": 268, "fbs": 0, "restecg": 0, "thalach": 152,
                "exang": 1, "oldpeak": 2.1, "slope": 1, "ca": 1, "thal": 2
            }
        }
    }


class BatchInput(BaseModel):
    patients: List[PatientInput] = Field(..., max_length=100,
                                         description="List of patients (max 100)")


class PredictionResponse(BaseModel):
    prediction:            int
    label:                 str
    risk_probability:      float
    risk_level:            str
    confidence:            float
    feature_contributions: dict
    recommendation:        str
    decision_threshold:    float
    latency_ms:            float
    model_version:         str
    timestamp:             str


class BatchPredictionResult(BaseModel):
    patient_index:    int
    prediction:       int
    risk_probability: float
    risk_level:       str


class BatchResponse(BaseModel):
    results:    List[BatchPredictionResult]
    errors:     List[dict]
    total_in:   int
    total_ok:   int
    total_err:  int
    latency_ms: float


class HealthResponse(BaseModel):
    status:    str
    model:     Optional[str]
    threshold: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_type:          Optional[str]
    n_estimators:        Optional[int]
    n_features:          Optional[int]
    feature_names:       Optional[list]
    feature_importance:  dict
    bundle_path:         str
    decision_threshold:  float


class MetricsResponse(BaseModel):
    total_requests:   int
    successful_preds: int
    failed_preds:     int
    high_risk_count:  int
    low_risk_count:   int
    start_time:       str
    uptime_seconds:   float
    success_rate:     float


# ── App ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("CardioSense AI FastAPI starting up...")
    yield
    log.info("CardioSense AI FastAPI shutting down...")

app = FastAPI(
    title="CardioSense AI",
    description=(
        "Heart disease prediction API using Random Forest with MLflow tracking.\n\n"
        "Built on the UCI Cleveland Heart Disease dataset (303 patients, 13 features).\n\n"
        "**Authors:** Arwin Dominic · Dinesh Kumar · Chethan K Chavan · Amaresh"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Helper ──────────────────────────────────────────────────────

def _get_risk(risk_prob: float, pred: int):
    if risk_prob >= 0.75:
        level = "HIGH"
        rec = (
            "Elevated cardiac risk detected. Immediate cardiologist referral "
            "is strongly advised. Request ECG, lipid panel, and stress test."
        )
    elif risk_prob >= 0.50:
        level = "MODERATE"
        rec = (
            "Moderate cardiac risk. Schedule a follow-up with your GP within "
            "2 weeks. Monitor blood pressure and cholesterol."
        )
    else:
        level = "LOW"
        rec = (
            "Low cardiac risk based on current values. Maintain a heart-healthy "
            "lifestyle: regular exercise, balanced diet, and annual health checks."
        )
    return level, rec


def _feature_contributions(top_n: int = 5) -> dict:
    if MODEL and hasattr(MODEL, "feature_importances_"):
        fi = dict(zip(FEATURE_COLS, MODEL.feature_importances_.round(4)))
        return dict(list(sorted(fi.items(), key=lambda x: x[1], reverse=True))[:top_n])
    return {}


# ── Routes ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — confirms model is loaded and API is responsive."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model=type(MODEL).__name__ if MODEL else None,
        threshold=THRESHOLD,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns model metadata, feature names, and feature importance scores."""
    fi = {}
    if MODEL and hasattr(MODEL, "feature_importances_"):
        fi = dict(zip(FEATURES, MODEL.feature_importances_.round(4).tolist()))
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    return ModelInfoResponse(
        model_type=type(MODEL).__name__ if MODEL else None,
        n_estimators=getattr(MODEL, "n_estimators", None),
        n_features=len(FEATURES) if FEATURES else None,
        feature_names=FEATURES,
        feature_importance=fi,
        bundle_path=BUNDLE_PATH,
        decision_threshold=THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Predict cardiac risk for a single patient.

    Returns prediction (0/1), risk probability, risk level (LOW/MODERATE/HIGH),
    clinical recommendation, and top feature contributions.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    STATS["total_requests"] += 1
    t0 = time.perf_counter()

    try:
        data = patient.model_dump()
        X = pd.DataFrame([data])[FEATURE_COLS]
        X_scaled = SCALER.transform(X)

        proba     = MODEL.predict_proba(X_scaled)[0]
        risk_prob = round(float(proba[1]), 4)
        pred      = int(risk_prob >= THRESHOLD)

        risk_level, recommendation = _get_risk(risk_prob, pred)
        top5 = _feature_contributions()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        STATS["successful_preds"] += 1
        if pred == 1:
            STATS["high_risk_count"] += 1
        else:
            STATS["low_risk_count"] += 1

        log.info(f"Prediction: {pred} | Risk: {risk_prob} | Level: {risk_level} | {latency_ms}ms")

        return PredictionResponse(
            prediction=pred,
            label="Heart Disease Detected" if pred == 1 else "No Heart Disease Detected",
            risk_probability=risk_prob,
            risk_level=risk_level,
            confidence=risk_prob if pred == 1 else round(1 - risk_prob, 4),
            feature_contributions=top5,
            recommendation=recommendation,
            decision_threshold=THRESHOLD,
            latency_ms=latency_ms,
            model_version="RandomForest_v2.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        STATS["failed_preds"] += 1
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """
    Batch prediction for up to 100 patients.

    Send a list of patient records and receive predictions for all of them.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    STATS["total_requests"] += 1
    t0 = time.perf_counter()

    results = []
    errors  = []

    for i, patient in enumerate(batch.patients):
        try:
            data     = patient.model_dump()
            X        = pd.DataFrame([data])[FEATURE_COLS]
            X_scaled = SCALER.transform(X)
            proba    = MODEL.predict_proba(X_scaled)[0]
            risk_prob = round(float(proba[1]), 4)
            pred     = int(risk_prob >= THRESHOLD)

            results.append(BatchPredictionResult(
                patient_index=i,
                prediction=pred,
                risk_probability=risk_prob,
                risk_level="HIGH" if risk_prob >= 0.75 else "MODERATE" if risk_prob >= 0.5 else "LOW",
            ))
            STATS["successful_preds"] += 1
        except Exception as e:
            errors.append({"patient_index": i, "error": str(e)})

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return BatchResponse(
        results=results,
        errors=errors,
        total_in=len(batch.patients),
        total_ok=len(results),
        total_err=len(errors),
        latency_ms=latency_ms,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics():
    """Runtime statistics — request counts, success rate, uptime."""
    uptime_s = (
        datetime.now(timezone.utc) -
        datetime.fromisoformat(STATS["start_time"])
    ).total_seconds()

    return MetricsResponse(
        **STATS,
        uptime_seconds=round(uptime_s, 1),
        success_rate=round(STATS["successful_preds"] / max(STATS["total_requests"], 1), 4),
    )


# ── Entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting CardioSense AI FastAPI on port {port}")
    log.info(f"Docs: http://0.0.0.0:{port}/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
