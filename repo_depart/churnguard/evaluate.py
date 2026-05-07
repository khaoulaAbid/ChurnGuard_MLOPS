"""Calcul des metriques de classification pour ChurnGuard."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calcule accuracy, precision, recall, f1 et roc_auc pour un modele entraine."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def build_classification_report(y_test: pd.Series, y_pred: pd.Series) -> str:
    """Construit un rapport de classification textuel pour les logs/artifacts."""
    return str(classification_report(y_test, y_pred))
