from math import exp
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Medical Risk Prediction API",
    version="1.0.0",
    description="Predicts a patient's relative risk from basic health indicators.",
)


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age in years")
    bmi: float = Field(..., ge=10, le=80, description="Body mass index")
    systolic_bp: int = Field(..., ge=70, le=250, description="Systolic blood pressure")
    glucose: float = Field(..., ge=40, le=500, description="Blood glucose (mg/dL)")
    cholesterol: float = Field(..., ge=50, le=500, description="Total cholesterol (mg/dL)")
    smoker: bool = Field(False, description="Whether the patient smokes")


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: Literal["low", "moderate", "high"]
    message: str


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def calculate_risk_probability(features: PatientFeatures) -> float:
    """Simple weighted-risk model (placeholder for a trained ML model)."""
    linear_score = (
        -7.5
        + 0.04 * features.age
        + 0.06 * (features.bmi - 21)
        + 0.015 * (features.systolic_bp - 120)
        + 0.02 * (features.glucose - 90)
        + 0.01 * (features.cholesterol - 170)
        + (0.8 if features.smoker else 0.0)
    )
    return round(_sigmoid(linear_score), 4)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    probability = calculate_risk_probability(features)

    if probability < 0.30:
        level = "low"
    elif probability < 0.70:
        level = "moderate"
    else:
        level = "high"

    return PredictionResponse(
        risk_probability=probability,
        risk_level=level,
        message="Prediction generated successfully.",
    )