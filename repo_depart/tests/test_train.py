"""Tests pour churnguard.train.train_model."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from churnguard import train
from churnguard.data import preprocess
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Mini DataFrame Telco pret a passer dans preprocess."""
    return pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5, 6],
            "MonthlyCharges": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "TotalCharges": [20.0, 60.0, 120.0, 200.0, 300.0, 420.0],
            "SeniorCitizen": [0, 1, 0, 1, 0, 1],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "Partner": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No"],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "No", "Yes"],
            "MultipleLines": [
                "No phone service",
                "No",
                "Yes",
                "No",
                "No phone service",
                "Yes",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "DSL",
                "DSL",
                "Fiber optic",
                "DSL",
            ],
            "OnlineSecurity": ["No", "No", "Yes", "No", "Yes", "No"],
            "DeviceProtection": ["No", "No", "Yes", "No", "Yes", "No"],
            "TechSupport": ["No", "No", "Yes", "No", "Yes", "No"],
            "StreamingTV": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "StreamingMovies": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "One year",
                "Month-to-month",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
            ],
            "Churn": ["No", "Yes", "No", "No", "Yes", "No"],
        }
    )


@pytest.fixture
def default_params() -> dict[str, Any]:
    """Hyperparametres rapides pour les tests."""
    return {
        "test_size": 0.34,
        "n_estimators": 10,
        "max_depth": 3,
        "max_iter": 200,
        "register_model": False,
    }


@pytest.fixture
def mock_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise toute interaction MLflow pour des tests offline."""

    class DummyRun:
        class Info:
            run_id = "dummy-run-id"

        info = Info()

        def __enter__(self) -> DummyRun:
            return self

        def __exit__(self, *_: Any) -> bool:
            return False

    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda *_: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda *_: None)
    monkeypatch.setattr(train.mlflow, "start_run", lambda **_: DummyRun())
    monkeypatch.setattr(train.mlflow, "log_params", lambda *_: None)
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda *_: None)
    monkeypatch.setattr(train.mlflow, "log_text", lambda *_: None)
    monkeypatch.setattr(train.mlflow, "log_dict", lambda *_: None)
    monkeypatch.setattr(train.mlflow, "set_tags", lambda *_: None)
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", lambda **_: None)


def test_train_model_returns_fitted_pipeline(
    sample_train_df: pd.DataFrame,
    default_params: dict[str, Any],
    mock_mlflow: None,
) -> None:
    """train_model retourne un Pipeline scikit-learn entraine et capable de predire."""
    X, y = preprocess(sample_train_df)

    model = train.train_model(X, y, "rf", default_params)

    assert isinstance(model, Pipeline)
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert set(y_pred).issubset({0, 1})


def test_train_model_supports_lr_and_gb(
    sample_train_df: pd.DataFrame,
    default_params: dict[str, Any],
    mock_mlflow: None,
) -> None:
    """train_model accepte aussi 'lr' et 'gb'."""
    X, y = preprocess(sample_train_df)

    for name in ["lr", "gb"]:
        model = train.train_model(X, y, name, default_params)
        assert isinstance(model, Pipeline)
        assert len(model.predict(X)) == len(y)


def test_train_model_rejects_unknown_model(
    sample_train_df: pd.DataFrame,
    default_params: dict[str, Any],
    mock_mlflow: None,
) -> None:
    """train_model leve ValueError pour un model_name inconnu."""
    X, y = preprocess(sample_train_df)

    with pytest.raises(ValueError):
        train.train_model(X, y, "bad", default_params)
