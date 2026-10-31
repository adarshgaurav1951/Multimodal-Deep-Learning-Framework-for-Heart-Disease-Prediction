"""
API Dependencies, Model State Manager, and Input Validation.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any
import numpy as np
import torch
from fastapi import HTTPException, status
from src.config import settings
from src.models.artifact import ModelArtifactManager
from src.api.schemas import PredictRequest


class ModelService:
    """Singleton holder for active model and processors."""
    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.clinical_processor = None
        self.ecg_processor = None
        self.threshold: float = 0.5
        self.version: str = "untrained"
        self.artifact_sha256: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.model_card: Dict[str, Any] = {}
        self.artifact_path: Optional[str] = None

    def load_from_path(self, path: str):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Model artifact not found at: {path}")

        bundle = ModelArtifactManager.load_artifact(path, device="cpu")
        self.model = bundle["model"]
        self.clinical_processor = bundle["clinical_processor"]
        self.ecg_processor = bundle["ecg_processor"]
        self.threshold = bundle["threshold"]
        self.version = bundle["version"]
        self.metrics = bundle["metrics"]
        self.model_card = bundle["model_card"]
        self.artifact_sha256 = bundle["sha256"]
        self.artifact_path = path

    def is_loaded(self) -> bool:
        return self.model is not None and self.clinical_processor is not None


model_service = ModelService()


def get_model_service() -> ModelService:
    # Attempt auto-loading if configured and not yet loaded
    if not model_service.is_loaded() and settings.MODEL_PATH and os.path.isfile(settings.MODEL_PATH):
        try:
            model_service.load_from_path(settings.MODEL_PATH)
        except Exception as e:
            # Keep unloaded; log warning
            print(f"Warning: Failed auto-loading model at {settings.MODEL_PATH}: {e}")
    return model_service


def validate_predict_payload(payload: PredictRequest) -> None:
    """Strict validation of incoming ECG time-series and clinical features."""
    if not payload.ecg or len(payload.ecg) not in {1, 12}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"ECG must contain exactly 1 or 12 leads, got {len(payload.ecg)}"
        )

    lengths = {len(lead) for lead in payload.ecg}
    if len(lengths) != 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"All ECG leads must have equal length, got mismatched lengths: {lengths}"
        )

    sample_count = next(iter(lengths))
    if sample_count < settings.MIN_ECG_SAMPLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"ECG length ({sample_count}) is below minimum required {settings.MIN_ECG_SAMPLES} samples"
        )

    if sample_count > settings.MAX_ECG_SAMPLES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"ECG length ({sample_count}) exceeds configured maximum {settings.MAX_ECG_SAMPLES}"
        )

    arr = np.asarray(payload.ecg, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ECG contains non-finite (NaN or Inf) values"
        )

    if not payload.clinical:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Clinical feature mapping cannot be empty"
        )

    for k, v in payload.clinical.items():
        if v is not None:
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    raise ValueError()
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Clinical feature '{k}' contains invalid non-numeric/non-finite value: {v}"
                )
