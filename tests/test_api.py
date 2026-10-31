"""
Integration tests for FastAPI endpoints, authentication, and error handling.
"""
import pytest
from fastapi.testclient import TestClient
from app import app
from src.database.session import init_db
from src.api.dependencies import model_service
from src.models.multimodal_classifier import MultimodalClassifier
from src.data.clinical_processor import ClinicalProcessor
from src.data.ecg_processor import ECGProcessor

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()


def get_auth_token(username="researcher@cardio.ai", password="Research2026!"):
    res = client.post(
        "/v1/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert res.status_code == 200
    return res.json()["access_token"]


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_model_metadata_disclaimer():
    res = client.get("/v1/model")
    assert res.status_code == 200
    data = res.json()
    assert "not medical diagnoses" in data["disclaimer"].lower() or "not a diagnosis" in data["disclaimer"].lower()


def test_auth_token_issuance():
    token = get_auth_token()
    assert len(token) > 20


def test_predict_requires_auth():
    payload = {
        "record_id": "REC_TEST_01",
        "sampling_rate_hz": 500.0,
        "ecg": [[0.0] * 500],
        "clinical": {"age": 55.0}
    }
    res = client.post("/v1/predict", json=payload)
    assert res.status_code == 401


def test_predict_rejects_invalid_lead_count():
    token = get_auth_token()
    payload = {
        "record_id": "REC_TEST_01",
        "sampling_rate_hz": 500.0,
        "ecg": [[0.0] * 500] * 5,  # 5 leads is invalid (only 1 or 12 allowed)
        "clinical": {"age": 55.0}
    }
    res = client.post("/v1/predict", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 422


def test_predict_rejects_short_ecg():
    token = get_auth_token()
    payload = {
        "record_id": "REC_TEST_01",
        "sampling_rate_hz": 500.0,
        "ecg": [[0.0] * 50],  # below min 100
        "clinical": {"age": 55.0}
    }
    res = client.post("/v1/predict", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 422


def test_predict_with_active_model():
    token = get_auth_token()

    # Configure mock active model on model_service singleton if not already loaded
    if not model_service.is_loaded():
        features = ["age", "sex", "resting_bp", "cholesterol", "max_hr"]
        clin_proc = ClinicalProcessor(feature_names=features)
        clin_proc.fit([{"age": 50, "sex": 1, "resting_bp": 120, "cholesterol": 200, "max_hr": 150}])
        model_service.model = MultimodalClassifier(in_leads=12, n_clinical_features=len(features), embedding_dim=64)
        model_service.clinical_processor = clin_proc
        model_service.ecg_processor = ECGProcessor(target_fs=500.0, nominal_samples=1000)
        model_service.version = "v1.0.0-test"
        model_service.threshold = 0.50

    payload = {
        "record_id": "REC_TEST_ACTIVE",
        "sampling_rate_hz": 500.0,
        "ecg": [[0.05] * 1000] * 12,
        "clinical": {"age": 62.0, "sex": 1.0, "resting_bp": 140.0, "cholesterol": 240.0, "max_hr": 135.0}
    }

    res = client.post("/v1/predict", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert "research_score" in data
    assert 0.0 <= data["research_score"] <= 1.0
    assert "disclaimer" in data
    assert "modality_contributions" in data
