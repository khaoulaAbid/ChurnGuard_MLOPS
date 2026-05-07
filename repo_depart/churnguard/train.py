"""Entrainement, evaluation et logging MLflow pour ChurnGuard."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from churnguard.data import build_preprocessor, cat_cols, load_data, num_cols, preprocess
from churnguard.evaluate import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_churn.csv")
ALL_FEATURES = num_cols + cat_cols

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "predictive-churn"
MODEL_REGISTRY_NAME = "churnguard"


def _build_classifier(model_name: str, params: dict[str, Any]) -> Any:
    """Instancie un classifier scikit-learn selon model_name."""
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "lr":
        return LogisticRegression(
            max_iter=params["max_iter"],
            random_state=42,
        )
    if model_name == "gb":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            random_state=42,
        )
    raise ValueError("model_name doit etre 'rf', 'lr' ou 'gb'")


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: dict[str, Any],
) -> Pipeline:
    """Entraine un Pipeline (preprocessor + classifier) et le logge dans MLflow."""
    logger.info("Dataset : %s lignes | Taux de churn : %.2f%%", len(X), float(y.mean()) * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=42,
        stratify=y,
    )
    logger.info("Train : %s | Test : %s", len(X_train), len(X_test))

    classifier = _build_classifier(model_name, params)
    preprocessor = build_preprocessor()
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"train_{model_name}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("MLflow Run ID : %s", run.info.run_id)

        mlflow.log_params(
            {
                "model_name": model_name,
                "n_estimators": params.get("n_estimators"),
                "max_depth": str(params.get("max_depth")),
                "max_iter": params.get("max_iter"),
                "test_size": params["test_size"],
                "n_features": len(ALL_FEATURES),
            }
        )

        model.fit(X_train, y_train)

        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        y_pred = model.predict(X_test)
        logger.info("Metriques test : %s", json.dumps(metrics))
        logger.info("\n%s", classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        cm_text = (
            f"Confusion Matrix\n{cm}\nTN={cm[0, 0]}  FP={cm[0, 1]}  FN={cm[1, 0]}  TP={cm[1, 1]}"
        )
        mlflow.log_text(cm_text, "confusion_matrix.txt")

        clf = model.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            fi_dict = dict(zip(ALL_FEATURES, clf.feature_importances_.tolist(), strict=False))
            mlflow.log_dict(fi_dict, "feature_importances.json")

        mlflow.set_tags(
            {
                "domain": "predictive-churn",
                "dataset": "telco_churn",
                "run_name": run_name,
            }
        )

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:3],
            signature=signature,
            registered_model_name=(MODEL_REGISTRY_NAME if params.get("register_model") else None),
        )

        logger.info(
            "Run termine (%s). f1_score=%.4f | Recall=%.4f | ROC-AUC=%.4f",
            model_name,
            metrics["f1_score"],
            metrics["recall"],
            metrics["roc_auc"],
        )

    return model


def _parse_args() -> argparse.Namespace:
    """Parse les arguments CLI pour un entrainement."""
    parser = argparse.ArgumentParser(description="Entrainement MLOps - Churn")
    parser.add_argument("--model", default="all", choices=["all", "rf", "lr", "gb"])
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logger.info("Chargement des donnees : %s", DATA_PATH)
    df = load_data(DATA_PATH)
    X, y = preprocess(df)

    params: dict[str, Any] = {
        "test_size": args.test_size,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "max_iter": args.max_iter,
        "register_model": args.register,
    }

    models_to_run = ["lr", "rf", "gb"] if args.model == "all" else [args.model]
    for model_name in models_to_run:
        train_model(X, y, model_name=model_name, params=params)
