from pathlib import Path

import pandas as pd
from churnguard.data import TARGET, cat_cols, load_data, num_cols, preprocess

EXPECTED_COLUMNS_AFTER_LOAD = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}  # 20 colonnes : les 21 brutes - customerID retire par load_data


def test_load_data_returns_dataframe(sample_csv: str) -> None:
    """load_data retourne bien un DataFrame de la bonne forme."""
    df = load_data(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 20)
    assert df.isna().sum().sum() == 0


def test_load_data_has_expected_columns(sample_csv: str) -> None:
    """load_data conserve les 20 colonnes attendues (21 brutes - customerID)."""
    df = load_data(sample_csv)

    assert set(df.columns) == EXPECTED_COLUMNS_AFTER_LOAD
    assert len(df.columns) == 20
    assert "customerID" not in df.columns
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_preprocess_returns_features_and_target(sample_csv: str) -> None:
    """preprocess separe correctement features (X) et cible binaire (y)."""
    df = load_data(sample_csv)
    X, y = preprocess(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == num_cols + cat_cols
    assert TARGET not in X.columns
    assert len(X) == len(y) == len(df)
    assert set(y.unique()).issubset({0, 1})


def test_preprocess_handles_missing_total_charges(
    sample_df_with_blank: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Les lignes avec TotalCharges vide sont supprimees apres nettoyage."""
    csv_path = tmp_path / "telco_blank.csv"
    sample_df_with_blank.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))
    X, y = preprocess(df)

    assert len(df) == 2
    assert len(X) == len(y) == 2
    assert df["TotalCharges"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
