"""Chargement paresseux du modele Production depuis le MLflow registry.

Compatible MLflow 2.x (stages classiques) et MLflow 3.x (aliases).
Par defaut on utilise l'alias `production` (MLflow 3.x).

Le modele est charge en mode sklearn (preserve `.predict_proba()`),
necessaire pour /predict qui renvoie une probabilite.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.sklearn
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
        self._client: MlflowClient | None = None

    def _resolve_version(self, client: MlflowClient) -> str:
        """Resout le numero de version a partir de l'alias, du stage ou de la derniere version."""
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            logger.info("Version resolved via alias '%s': v%s", MODEL_ALIAS, mv.version)
            return str(mv.version)
        except Exception as exc_alias:
            logger.warning("Cannot resolve via alias '%s': %s", MODEL_ALIAS, exc_alias)

        try:
            latest = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if latest:
                logger.info("Version resolved via stage Production: v%s", latest[0].version)
                return str(latest[0].version)
        except Exception as exc_stage:
            logger.warning("Cannot resolve via stage Production: %s", exc_stage)

        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if versions:
                latest_version = max(versions, key=lambda v: int(v.version))
                logger.info("Version resolved via latest fallback: v%s", latest_version.version)
                return str(latest_version.version)
        except Exception as exc_search:
            logger.warning("Cannot resolve via search: %s", exc_search)

        return "unknown"

    def _resolve_target_run_id(self, client: MlflowClient) -> str | None:
        """Resout le run_id de la version cible (alias -> stage -> latest)."""
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            return str(mv.run_id)
        except Exception:
            pass

        try:
            latest = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if latest:
                return str(latest[0].run_id)
        except Exception:
            pass

        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if versions:
                target = max(versions, key=lambda v: int(v.version))
                return str(target.run_id)
        except Exception:
            pass
        return None

    def load(self) -> None:
        """Charge le modele MLflow et resout la version associee."""
        self.model = None
        self.version = "unknown"
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            self._client = client
            try:
                self.model = mlflow.sklearn.load_model(MODEL_URI)
            except Exception as exc_primary:
                logger.warning("Primary model load failed from %s: %s", MODEL_URI, exc_primary)
                run_id = self._resolve_target_run_id(client)
                if run_id is None:
                    raise

                loaded = False
                fallback_uris = [f"runs:/{run_id}/model"]
                try:
                    artifact_uri = client.get_run(run_id).info.artifact_uri.rstrip("/")
                    fallback_uris.append(f"{artifact_uri}/model")
                except Exception as exc_artifact:
                    logger.warning("Cannot resolve artifact_uri for run %s: %s", run_id, exc_artifact)

                for uri in fallback_uris:
                    try:
                        self.model = mlflow.sklearn.load_model(uri)
                        logger.info("Model loaded via fallback URI: %s", uri)
                        loaded = True
                        break
                    except Exception as exc_fallback:
                        logger.warning("Fallback load failed for %s: %s", uri, exc_fallback)

                if not loaded:
                    raise RuntimeError(
                        f"Unable to load model from primary or fallback URIs (run_id={run_id})"
                    ) from exc_primary
            self.version = self._resolve_version(client)
            logger.info("Model loaded: %s (version=%s)", MODEL_URI, self.version)
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", MODEL_URI, exc)
            self.model = None
            self.version = "unknown"
            self._client = None

    def refresh_version(self) -> str:
        """Rafraichit la version depuis le registry sans recharger le modele."""
        if self._client is None:
            try:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                self._client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            except Exception as exc:
                logger.warning("Cannot initialize MlflowClient for refresh: %s", exc)
                return self.version

        try:
            resolved = self._resolve_version(self._client)
            if resolved != "unknown":
                self.version = resolved
        except Exception as exc:
            logger.warning("Cannot refresh model version: %s", exc)
        return self.version

    @property
    def is_loaded(self) -> bool:
        """Indique si un modele est disponible en memoire."""
        return self.model is not None


model_loader = ModelLoader()
