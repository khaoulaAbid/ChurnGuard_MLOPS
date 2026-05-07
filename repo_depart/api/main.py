"""FastAPI inference service backed by MLflow registry model."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from api.model_loader import model_loader
from api.schemas import BatchPredictRequest, ChurnFeatures


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup from MLflow registry."""
    model_loader.load()
    yield


app = FastAPI(title="ChurnGuard API", version="1.0.0", lifespan=lifespan)


def _ensure_model_loaded() -> None:
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health")
def health() -> dict[str, str]:
    """Return health and loaded model metadata."""
    return {
        "status": "ok",
        "model": "churnguard",
        "version": model_loader.version,
    }


@app.post("/predict")
def predict(payload: ChurnFeatures) -> dict[str, float | bool]:
    """Predict churn for one customer."""
    _ensure_model_loaded()
    frame = pd.DataFrame([payload.model_dump()])
    prediction = model_loader.model.predict(frame)
    probability = model_loader.model.predict_proba(frame)[:, 1]
    return {"churn": bool(int(prediction[0])), "probability": float(probability[0])}


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> dict[str, list[dict[str, float | bool]]]:
    """Predict churn for a batch of customers."""
    _ensure_model_loaded()
    if len(payload.records) == 0:
        raise HTTPException(status_code=400, detail="Batch cannot be empty")
    if len(payload.records) > 100:
        raise HTTPException(status_code=400, detail="Batch size must be <= 100")
    frame = pd.DataFrame([item.model_dump() for item in payload.records])
    predictions = model_loader.model.predict(frame)
    probabilities = model_loader.model.predict_proba(frame)[:, 1]
    results = [
        {"churn": bool(int(pred)), "probability": float(prob)}
        for pred, prob in zip(predictions, probabilities, strict=True)
    ]
    return {"predictions": results}

