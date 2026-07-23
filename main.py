from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "models/xgboost_model.joblib"

# Populated once at startup, reused across every request.
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when the server boots — never inside a request handler.
    # Loading a ~few-MB pickle on every request would tank latency.
    ml_models["churn_pipeline"] = joblib.load(MODEL_PATH)
    yield
    ml_models.clear()


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


class CustomerInput(BaseModel):
    """Schema matching the raw IBM Telco Customer Churn features.
    customerID, MonthlyCharges, and TotalCharges are required here even
    though the model never uses them — the ColumnTransformer's 'drop'
    step still needs them present in the input to validate against."""

    customerID: str
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    customer_id: str
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float


@app.get("/health")
def health_check():
    """Cheap endpoint for AWS/monitoring to confirm the service is alive."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    pipeline = ml_models.get("churn_pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Pipeline expects a DataFrame with the exact training column names —
    # wrapping the single dict in a list gives us a one-row DataFrame.
    input_df = pd.DataFrame([customer.model_dump()])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    return PredictionResponse(
        customer_id=customer.customerID,
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(float(probability), 4),
    )