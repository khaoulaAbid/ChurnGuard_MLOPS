"""Chargement et pre-processing du dataset Telco Customer Churn."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols: list[str] = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET: str = "Churn"


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV Telco et applique un nettoyage de base."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separe les features (X) et la cible binaire (y) a partir du DataFrame."""
    all_features = num_cols + cat_cols
    X = df[all_features]
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Construit un ColumnTransformer (scaling + one-hot) pour les colonnes definies."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )
