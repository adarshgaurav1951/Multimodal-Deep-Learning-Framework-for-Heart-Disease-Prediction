"""
FastAPI Route Handlers.
Implements authentication, model registry inspection, inference, explainability, and audit logging.
"""
from __future__ import annotations

import time
from typing import Dict, Any, List
import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.config import settings
from src.database.session import get_db
from src.database.service import DatabaseService
from src.security.auth import (
    verify_password,
    create_access_token,
    get_current_user,
    require_role
)
from src.security.crypto import generate_request_id
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainRequest,
    ExplainResponse,
    TokenResponse,
    RegisterUserRequest,
    UserResponse,
    ModelMetadataResponse,
    LoadModelRequest
)
from src.api.dependencies import get_model_service, ModelService, validate_predict_payload
from src.models.explainability import compute_ecg_saliency, compute_clinical_attributions
from src.data.synthetic_generator import generate_12lead_ecg

api_router = APIRouter()


@api_router.get("/health", tags=["System"])
def health(ms: ModelService = Depends(get_model_service)):
    """Liveness probe and model deployment status."""
    return {
        "status": "ok",
        "model_loaded": ms.is_loaded(),
        "model_version": ms.version if ms.is_loaded() else settings.MODEL_VERSION,
        "app_env": settings.APP_ENV
    }


@api_router.post("/v1/auth/token", response_model=TokenResponse, tags=["Authentication"])
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """OAuth2 password login returning signed JWT."""
    user = DatabaseService.get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user account")

    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    DatabaseService.log_audit(db, action="login_success", actor_id=user.id, actor_username=user.email)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in_minutes": settings.ACCESS_TOKEN_MINUTES,
        "role": user.role
    }


@api_router.post("/v1/auth/register", response_model=UserResponse, tags=["Authentication"])
def register_user(
    payload: RegisterUserRequest,
    db: Session = Depends(get_db)
):
    """Register a new user account (defaults to researcher role)."""
    existing = DatabaseService.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is already registered")

    valid_roles = {"admin", "researcher", "reviewer", "api_client"}
    role = payload.role if payload.role in valid_roles else "researcher"
    user = DatabaseService.create_user(db, email=payload.email, password=payload.password, role=role)
    DatabaseService.log_audit(db, action="user_registered", actor_id=user.id, actor_username=user.email)
    return user


@api_router.get("/v1/model", response_model=ModelMetadataResponse, tags=["Model Registry"])
def get_model_metadata(ms: ModelService = Depends(get_model_service)):
    """Retrieve metadata, parameters, model card, and clinical disclaimers for active model."""
    return {
        "model_version": ms.version,
        "model_configured": ms.is_loaded(),
        "artifact_path": ms.artifact_path,
        "artifact_sha256": ms.artifact_sha256,
        "threshold": ms.threshold if ms.is_loaded() else None,
        "metrics": ms.metrics,
        "model_card": ms.model_card,
        "status": "active" if ms.is_loaded() else "no_model_configured",
        "disclaimer": "Research prototype only. Outputs are experimental model scores, not medical diagnoses or clinical advice."
    }


@api_router.post("/v1/model/load", tags=["Model Registry"])
def load_model_artifact(
    payload: LoadModelRequest,
    ms: ModelService = Depends(get_model_service),
    user: Dict[str, Any] = Depends(require_role("admin", "researcher")),
    db: Session = Depends(get_db)
):
    """Hot-swap or load an approved model artifact from disk."""
    try:
        ms.load_from_path(payload.artifact_path)
        DatabaseService.log_audit(
            db,
            action="model_artifact_loaded",
            actor_username=user["username"],
            metadata={"artifact_path": payload.artifact_path, "sha256": ms.artifact_sha256}
        )
        return {
            "status": "success",
            "message": f"Successfully loaded model version {ms.version}",
            "sha256": ms.artifact_sha256,
            "threshold": ms.threshold
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed loading model artifact: {str(e)}")


@api_router.post("/v1/predict", response_model=PredictResponse, tags=["Inference"])
def predict(
    payload: PredictRequest,
    ms: ModelService = Depends(get_model_service),
    user: Dict[str, Any] = Depends(require_role("admin", "researcher", "api_client")),
    db: Session = Depends(get_db)
):
    """
    Multimodal cardiovascular risk inference endpoint.
    Performs input validation, Butterworth signal filtering, clinical normalization,
    and neural fusion scoring with audit telemetry.
    """
    start_time = time.perf_counter()
    request_id = generate_request_id()

    # Strict input validation
    validate_predict_payload(payload)

    # Fail closed if no model is loaded
    if not ms.is_loaded():
        DatabaseService.log_inference(db, request_id=request_id, outcome="model_not_loaded", latency_ms=0)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained model artifact is loaded. Train and register an artifact first."
        )

    try:
        # 1. Preprocess ECG
        proc_ecg = ms.ecg_processor.process(payload.ecg, orig_fs=payload.sampling_rate_hz)
        ecg_tensor = torch.from_numpy(proc_ecg).unsqueeze(0).float()  # [1, leads, samples]

        # 2. Preprocess Clinical
        proc_clinical = ms.clinical_processor.transform(payload.clinical)
        clinical_tensor = torch.from_numpy(proc_clinical).unsqueeze(0).float()  # [1, num_features]

        # 3. Model forward pass
        with torch.no_grad():
            logits, details = ms.model(ecg_tensor, clinical_tensor, return_details=True)
            score = float(torch.sigmoid(logits).item())

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        label = "Elevated Risk Indication (Experimental)" if score >= ms.threshold else "Low/Baseline Risk Indication (Experimental)"

        # Modality contribution extraction from gating mechanism
        fusion_meta = details.get("fusion_meta", {})
        ecg_w = fusion_meta.get("ecg_weight", 0.5) if isinstance(fusion_meta, dict) else 0.5
        clin_w = fusion_meta.get("clinical_weight", 0.5) if isinstance(fusion_meta, dict) else 0.5

        # Audit log minimal telemetry (no raw PHI logged)
        DatabaseService.log_inference(
            db,
            request_id=request_id,
            outcome="success",
            latency_ms=latency_ms
        )

        return {
            "request_id": request_id,
            "model_version": ms.version,
            "research_score": round(score, 4),
            "decision_threshold": round(ms.threshold, 4),
            "prediction_label": label,
            "modality_contributions": {
                "ecg_weight": round(ecg_w, 3),
                "clinical_weight": round(clin_w, 3)
            },
            "latency_ms": latency_ms,
            "disclaimer": "Research prototype only. Outputs are experimental model scores, not medical diagnoses or clinical advice."
        }
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        DatabaseService.log_inference(db, request_id=request_id, outcome="error", latency_ms=latency_ms)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(e)}"
        )


@api_router.post("/v1/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain(
    payload: ExplainRequest,
    ms: ModelService = Depends(get_model_service),
    user: Dict[str, Any] = Depends(require_role("admin", "researcher")),
    db: Session = Depends(get_db)
):
    """
    Compute Integrated Gradients saliency on ECG waveforms and feature attribution on clinical variables.
    """
    request_id = generate_request_id()
    validate_predict_payload(payload)

    if not ms.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact must be loaded to generate explanations."
        )

    # Preprocess
    proc_ecg = ms.ecg_processor.process(payload.ecg, orig_fs=payload.sampling_rate_hz)
    ecg_tensor = torch.from_numpy(proc_ecg).unsqueeze(0).float()
    proc_clinical = ms.clinical_processor.transform(payload.clinical)
    clinical_tensor = torch.from_numpy(proc_clinical).unsqueeze(0).float()

    # Score
    with torch.no_grad():
        score = float(torch.sigmoid(ms.model(ecg_tensor, clinical_tensor)).item())

    # ECG Saliency
    ecg_saliency = compute_ecg_saliency(ms.model, ecg_tensor, clinical_tensor, steps=payload.steps)

    # Subsample saliency matrix for light JSON payload (e.g. 1 sample every 10)
    mat = np.array(ecg_saliency["saliency_matrix"])
    subsampled = mat[:, ::10].tolist()

    # Clinical attributions
    clin_attr = compute_clinical_attributions(
        ms.model,
        ecg_tensor,
        clinical_tensor,
        feature_names=ms.clinical_processor.feature_names
    )

    return {
        "request_id": request_id,
        "research_score": round(score, 4),
        "lead_ranking": ecg_saliency["lead_ranking"],
        "clinical_attributions": clin_attr,
        "saliency_preview": subsampled,
        "method": ecg_saliency["method"],
        "disclaimer": "Explanations indicate correlation with model scoring and are not proof of physiological causality."
    }


@api_router.get("/v1/experiments", tags=["Research"])
def list_experiments(
    limit: int = 10,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(require_role("admin", "researcher", "reviewer"))
):
    """Retrieve logged experiments and validation benchmarks."""
    exps = DatabaseService.list_experiments(db, limit=limit)
    res = []
    for exp in exps:
        metrics_dict = {m.split_name: m.values for m in exp.metrics}
        res.append({
            "id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "config": exp.config,
            "metrics": metrics_dict,
            "created_at": exp.created_at.isoformat() if exp.created_at else None
        })
    return res


@api_router.get("/v1/audit", tags=["Governance"])
def list_audit_logs(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(require_role("admin"))
):
    """Retrieve system audit logs (Admin only)."""
    logs = DatabaseService.list_audit_logs(db, limit=limit)
    return [
        {
            "id": log.id,
            "action": log.action,
            "actor_username": log.actor_username,
            "resource_type": log.resource_type,
            "request_id": log.request_id,
            "metadata": log.metadata_json,
            "created_at": log.created_at.isoformat() if log.created_at else None
        }
        for log in logs
    ]


@api_router.get("/v1/sample-record", tags=["Sandbox"])
def get_sample_record(pathological: bool = True):
    """
    Generate a realistic synthetic 12-lead ECG and clinical profile for sandbox exploration.
    """
    ecg = generate_12lead_ecg(duration_sec=5.0, fs=500.0, hr=85.0 if pathological else 68.0, is_pathological=pathological)
    clinical = {
        "age": 64.0 if pathological else 42.0,
        "sex": 1.0,
        "resting_bp": 155.0 if pathological else 118.0,
        "cholesterol": 268.0 if pathological else 185.0,
        "fasting_bs": 1.0 if pathological else 0.0,
        "resting_ecg": 2.0 if pathological else 0.0,
        "max_hr": 128.0 if pathological else 165.0,
        "exercise_angina": 1.0 if pathological else 0.0,
        "st_depression": 2.4 if pathological else 0.2,
        "st_slope": 1.0 if pathological else 0.0,
        "num_major_vessels": 2.0 if pathological else 0.0,
        "thal": 2.0 if pathological else 0.0
    }
    return {
        "record_id": f"SAMPLE_{'HIGH_RISK' if pathological else 'NORMAL'}",
        "sampling_rate_hz": 500.0,
        "ecg": ecg.tolist(),
        "clinical": clinical
    }
