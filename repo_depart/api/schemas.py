"""Pydantic schemas for ChurnGuard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChurnFeatures(BaseModel):
    """Validated telco customer features for inference."""

    model_config = ConfigDict(extra="forbid", strict=True)

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=200)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0, le=10000.0)
    TotalCharges: float = Field(ge=0.0, le=1000000.0)


class BatchPredictRequest(BaseModel):
    """Validated batch payload for inference."""

    model_config = ConfigDict(extra="forbid", strict=True)
    records: list[ChurnFeatures]

