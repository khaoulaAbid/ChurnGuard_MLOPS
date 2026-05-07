from pathlib import Path

import pytest
from churnguard.data import load_data

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"


@pytest.mark.integration
@pytest.mark.skipif(not REAL_CSV.exists(), reason="CSV reel absent")
def test_load_data_on_real_csv() -> None:
    """Smoke test sur le vrai CSV Telco (skip si absent)."""
    df = load_data(str(REAL_CSV))
    assert df.shape[1] == 20
    assert df.shape[0] > 1000
