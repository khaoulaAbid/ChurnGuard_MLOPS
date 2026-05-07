import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Mini DataFrame imitant la structure du CSV Telco (3 lignes, 21 colonnes)."""
    return pd.DataFrame(
        {
            "customerID": ["1", "2", "3"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [1, 2, 3],
            "PhoneService": ["No", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "Yes"],
            "DeviceProtection": ["No", "No", "Yes"],
            "TechSupport": ["No", "No", "Yes"],
            "StreamingTV": ["No", "Yes", "No"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": ["29.85", "56.95", "108.15"],
            "Churn": ["No", "Yes", "No"],
        }
    )


@pytest.fixture
def sample_df_with_blank() -> pd.DataFrame:
    """Variante avec une cellule TotalCharges vide (pour tester le nettoyage)."""
    df = pd.DataFrame(
        {
            "customerID": ["1", "2", "3"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [1, 2, 3],
            "PhoneService": ["No", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "Yes"],
            "DeviceProtection": ["No", "No", "Yes"],
            "TechSupport": ["No", "No", "Yes"],
            "StreamingTV": ["No", "Yes", "No"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": ["29.85", " ", "108.15"],
            "Churn": ["No", "Yes", "No"],
        }
    )
    return df


@pytest.fixture
def sample_csv(sample_df: pd.DataFrame, tmp_path: Path) -> str:
    """Ecrit sample_df dans un fichier temporaire et renvoie son chemin."""
    csv_path = tmp_path / "telco.csv"
    sample_df.to_csv(csv_path, index=False)
    return str(csv_path)