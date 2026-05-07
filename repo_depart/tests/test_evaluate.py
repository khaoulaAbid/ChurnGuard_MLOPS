import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from churnguard.evaluate import compute_metrics


def test_compute_metrics_returns_expected_keys() -> None:
    X_train = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1],
            "f2": [1, 1, 0, 0, 1, 0],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    X_test = pd.DataFrame(
        {
            "f1": [0, 1, 1],
            "f2": [1, 0, 1],
        }
    )
    y_test = pd.Series([0, 1, 1])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    assert set(metrics.keys()) == expected_keys
    assert len(metrics) == 5
