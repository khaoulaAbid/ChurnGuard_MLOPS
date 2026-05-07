"""Chargement paresseux du modele Production depuis le MLflow registry."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "models:/churnguard/Production")
MODEL_NAME = os.getenv("MODEL_NAME", "churnguard")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class ModelLoader:
    """Charge et expose le modele Production depuis le MLflow registry."""

    def __init__(self) -> None:
        self.model: Any = None
        self.version: str = "unknown"

    def load(self) -> None:
        """Charge le modele MLflow et resout la version Production."""
        self.model = None
        self.version = "unknown"
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            self.model = mlflow.pyfunc.load_model(MODEL_URI)
            latest = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if latest:
                self.version = str(latest[0].version)
            logger.info("Model loaded: %s (version=%s)", MODEL_URI, self.version)
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", MODEL_URI, exc)
            self.model = None
            self.version = "unknown"

    @property
    def is_loaded(self) -> bool:
        """Indique si un modele est disponible en memoire."""
        return self.model is not None


model_loader = ModelLoader()
