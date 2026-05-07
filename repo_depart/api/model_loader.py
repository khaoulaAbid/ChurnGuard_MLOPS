"""Chargement paresseux du modele Production depuis le MLflow registry.

Compatible MLflow 2.x (stages classiques) et MLflow 3.x (aliases).
Par defaut on utilise l'alias `production` (MLflow 3.x).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "models:/churnguard@production")
MODEL_NAME = os.getenv("MODEL_NAME", "churnguard")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class ModelLoader:
    """Charge et expose le modele Production depuis le MLflow registry."""

    def __init__(self) -> None:
        self.model: Any = None
        self.version: str = "unknown"

    def _resolve_version(self, client: MlflowClient) -> str:
        """Resout le numero de version a partir de l'alias ou du stage."""
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            return str(mv.version)
        except Exception as exc_alias:
            logger.debug("alias '%s' indisponible : %s", MODEL_ALIAS, exc_alias)
        try:
            latest = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if latest:
                return str(latest[0].version)
        except Exception as exc_stage:
            logger.debug("stage 'Production' indisponible : %s", exc_stage)
        return "unknown"

    def load(self) -> None:
        """Charge le modele MLflow et resout la version associee."""
        self.model = None
        self.version = "unknown"
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            self.model = mlflow.pyfunc.load_model(MODEL_URI)
            self.version = self._resolve_version(client)
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
