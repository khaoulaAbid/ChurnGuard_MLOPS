from argparse import Namespace

import pandas as pd
from sklearn.pipeline import Pipeline

from churnguard import train


def _mock_mlflow(monkeypatch) -> None:
    class DummyRun:
        class Info:
            run_id = "dummy-run-id"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
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


def _sample_frame() -> pd.DataFrame:
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
            "MultipleLines": ["No phone service", "No", "Yes", "No", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No", "Yes", "No", "Yes", "No"],
            "DeviceProtection": ["No", "No", "Yes", "No", "Yes", "No"],
            "TechSupport": ["No", "No", "Yes", "No", "Yes", "No"],
            "StreamingTV": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "StreamingMovies": ["No", "Yes", "No", "Yes", "No", "Yes"],
            "Contract": ["Month-to-month", "One year", "Two year", "One year", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Mailed check", "Electronic check", "Bank transfer (automatic)"],
            "Churn": ["No", "Yes", "No", "No", "Yes", "No"],
        }
    )


def test_train_model_returns_fitted_pipeline(monkeypatch) -> None:
    sample = pd.DataFrame(_sample_frame())

    monkeypatch.setattr(train, "load_data", lambda _path: sample.copy())
    _mock_mlflow(monkeypatch)
    params = Namespace(
        test_size=0.3,
        n_estimators=10,
        max_depth=3,
        max_iter=200,
        register_model=False,
    )

    model, X_test, y_test, y_pred, y_proba = train.train_model("rf", params)

    assert isinstance(model, Pipeline)
    assert len(y_pred) == len(y_test)
    assert len(y_proba) == len(y_test)


def test_train_model_supports_lr_and_gb(monkeypatch) -> None:
    monkeypatch.setattr(train, "load_data", lambda _path: _sample_frame().copy())
    _mock_mlflow(monkeypatch)
    params = Namespace(
        test_size=0.3,
        n_estimators=10,
        max_depth=3,
        max_iter=200,
        register_model=False,
    )

    for name in ["lr", "gb"]:
        model, X_test, y_test, y_pred, _ = train.train_model(name, params)
        assert isinstance(model, Pipeline)
        assert len(X_test) == len(y_test)
        assert len(y_pred) == len(y_test)


def test_train_model_rejects_unknown_model(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(train, "load_data", lambda _path: _sample_frame().copy())
    _mock_mlflow(monkeypatch)
    params = Namespace(
        test_size=0.3,
        n_estimators=10,
        max_depth=3,
        max_iter=200,
        register_model=False,
    )

    with pytest.raises(ValueError):
        train.train_model("bad", params)