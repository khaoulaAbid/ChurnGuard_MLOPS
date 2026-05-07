"""Tests pour churnguard.evaluate.compute_metrics."""

from __future__ import annotations

import pandas as pd
from churnguard.evaluate import compute_metrics
from sklearn.ensemble import RandomForestClassifier


def test_compute_metrics_returns_expected_keys() -> None:
    """compute_metrics renvoie un dict avec les 5 metriques attendues."""
    X_train = pd.DataFrame({"f1": [0, 1, 0, 1, 0, 1], "f2": [1, 1, 0, 0, 1, 0]})
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame({"f1": [0, 1, 1], "f2": [1, 0, 1]})
    y_test = pd.Series([0, 1, 1])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    metrics = compute_metrics(model, X_test, y_test)

    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    assert set(metrics.keys()) == expected_keys
    assert len(metrics) == 5
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
