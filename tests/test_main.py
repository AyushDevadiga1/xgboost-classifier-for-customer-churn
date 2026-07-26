"""
Smoke tests for the churn prediction API. Run with `pytest` from the
project root. These exist mainly to give the CI/CD pipeline something
real to check before it builds and pushes an image — not exhaustive
coverage of every edge case.
"""

import os

# Must be set BEFORE `main` is imported — API_KEY is read at module load time.
os.environ["API_KEY"] = "test-key-for-ci"

import pytest
from fastapi.testclient import TestClient

from main import app

VALID_CUSTOMER = {
    "customerID": "0000-TEST",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
}


@pytest.fixture()
def client():
    # Using the context-manager form ensures the `lifespan` block actually
    # runs (loading the real pipeline) instead of being skipped.
    with TestClient(app) as c:
        yield c


def test_health_check_is_unauthenticated(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_missing_api_key(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 401


def test_predict_rejects_wrong_api_key(client):
    response = client.post(
        "/predict", json=VALID_CUSTOMER, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


def test_predict_returns_valid_shape_with_correct_key(client):
    response = client.post(
        "/predict", json=VALID_CUSTOMER, headers={"X-API-Key": "test-key-for-ci"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "0000-TEST"
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_malformed_input(client):
    bad_customer = dict(VALID_CUSTOMER)
    bad_customer["InternetService"] = "Cable"  # not a valid schema value
    response = client.post(
        "/predict", json=bad_customer, headers={"X-API-Key": "test-key-for-ci"}
    )
    assert response.status_code == 422
