"""Lazy loader for MLflow production model."""

from __future__ import annotations

import os

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

MODEL_URI = os.getenv("MODEL_URI", "models:/churnguard/Production")
MODEL_NAME = os.getenv("MODEL_NAME", "churnguard")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class ModelLoader:
    """Load and expose production model from MLflow registry."""

    def __init__(self) -> None:
        self.model = None
        self.version = "unknown"

    def load(self) -> None:
        """Load MLflow model and resolve production version."""
        self.model = None
        self.version = "unknown"
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            self.model = mlflow.pyfunc.load_model(MODEL_URI)
            latest = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if latest:
                self.version = str(latest[0].version)
        except Exception:
            self.model = None
            self.version = "unknown"

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is available."""
        return self.model is not None


model_loader = ModelLoader()

