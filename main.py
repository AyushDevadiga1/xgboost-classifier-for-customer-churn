"""
FastAPI wrapper around the existing customer-churn prediction pipeline.

CONFIRMED (from models/xgboost_model.joblib + the training notebook):
this is a single fitted sklearn Pipeline — a ColumnTransformer feeding an
XGBClassifier. The ColumnTransformer does four things to the 20 raw
columns: binary-maps 12 Yes/No-style columns, ordinal-encodes Contract,
target-encodes InternetService/PaymentMethod, and DROPS customerID,
MonthlyCharges, and TotalCharges. Because sklearn's ColumnTransformer
validates that every referenced column exists at transform time — even
ones mapped to 'drop' — all three of those still have to be present in
the request payload, or `.predict()` raises a KeyError.

Also includes: request logging, and an API key requirement on /predict.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Looks unused — it isn't. The pickled pipeline stores a reference to
# to_binary_map by module path (src.binary_map), not its actual code.
# joblib.load() re-imports it at load time; without this package present
# and importable, loading the model raises ModuleNotFoundError.
import src.binary_map  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("churn_api")

MODEL_PATH = "models/xgboost_model.joblib"

# Set at runtime via `docker run -e API_KEY=...` — never baked into the image.
API_KEY = os.environ.get("API_KEY")

# Populated once at startup, reused across every request.
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when the server boots — never inside a request handler.
    # Loading a pickle on every request would tank latency.
    if not API_KEY:
        logger.warning(
            "API_KEY is not set — every /predict request will be rejected until it is."
        )
    ml_models["churn_pipeline"] = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
    yield
    ml_models.clear()


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str = Security(api_key_header)):
    """Dependency guarding /predict. /health stays open on purpose —
    load balancers and uptime checks need to poll it without a key."""
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key not configured on server")
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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
    """Cheap, unauthenticated endpoint for AWS/monitoring to confirm the service is alive."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(customer: CustomerInput):
    pipeline = ml_models.get("churn_pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info("Prediction requested for customer_id=%s", customer.customerID)

    # Pipeline expects a DataFrame with the exact training column names —
    # wrapping the single dict in a list gives us a one-row DataFrame.
    input_df = pd.DataFrame([customer.model_dump()])

    try:
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
    except Exception:
        logger.exception("Prediction failed for customer_id=%s", customer.customerID)
        raise HTTPException(status_code=500, detail="Prediction failed")

    result = "Yes" if prediction == 1 else "No"
    logger.info(
        "customer_id=%s prediction=%s probability=%.4f",
        customer.customerID,
        result,
        probability,
    )

    return PredictionResponse(
        customer_id=customer.customerID,
        churn_prediction=result,
        churn_probability=round(float(probability), 4),
    )
