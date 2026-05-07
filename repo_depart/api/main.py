"""Service FastAPI d'inference adosse au Model Registry MLflow."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import model_loader
from api.schemas import BatchPredictRequest, ChurnFeatures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modele une fois au demarrage depuis le registry MLflow."""
    model_loader.load()
    yield


app = FastAPI(title="ChurnGuard API", version="1.0.0", lifespan=lifespan)


def _ensure_model_loaded() -> None:
    """Leve une 503 si le modele n'a pas pu etre charge."""
    if not model_loader.is_loaded:
        model_loader.load()
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health")
def health() -> dict[str, str]:
    """Renvoie le statut du service et la version du modele charge."""
    # Le modele peut etre promu apres le demarrage de l'API
    # On rafraichit donc la version depuis le registry a chaque healthcheck
    model_loader.refresh_version()
    return {
        "status": "ok",
        "model": "churnguard",
        "version": model_loader.version,
    }


@app.post("/predict")
def predict(payload: ChurnFeatures) -> dict[str, Any]:
    """Predit le churn pour un client unique."""
    _ensure_model_loaded()
    frame = pd.DataFrame([payload.model_dump()])
    prediction = model_loader.model.predict(frame)
    probability = model_loader.model.predict_proba(frame)[:, 1]
    return {
        "churn": bool(int(prediction[0])),
        "probability": float(probability[0]),
    }


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> dict[str, list[dict[str, Any]]]:
    """Predit le churn pour un lot de clients (1 a 100)."""
    _ensure_model_loaded()
    if len(payload.records) == 0:
        raise HTTPException(status_code=400, detail="Batch cannot be empty")
    if len(payload.records) > 100:
        raise HTTPException(status_code=400, detail="Batch size must be <= 100")
    frame = pd.DataFrame([item.model_dump() for item in payload.records])
    predictions = model_loader.model.predict(frame)
    probabilities = model_loader.model.predict_proba(frame)[:, 1]
    results: list[dict[str, Any]] = [
        {"churn": bool(int(pred)), "probability": float(prob)}
        for pred, prob in zip(predictions, probabilities, strict=True)
    ]
    return {"predictions": results}
