from fastapi import params
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "Churn"

# Chargement
def load_data(path: str) -> pd.DataFrame:
    """Chargement du dataset telco_churn """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    return df

# Preprocessing
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les features numériques/catégorielles et la cible binaire."""

    all_features = num_cols + cat_cols
    X = df[all_features]
    y = (df[TARGET] == "Yes").astype(int)

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Construit un ColumnTransformer (scaling + one-hot) pour les colonnes définies."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )