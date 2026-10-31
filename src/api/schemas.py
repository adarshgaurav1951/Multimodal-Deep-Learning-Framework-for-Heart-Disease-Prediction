"""
Pydantic Request and Response Schemas for the FastAPI REST Service.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(..., min_length=1, max_length=128, description="Pseudonymous research record ID")
    sampling_rate_hz: float = Field(500.0, gt=0, le=5000, description="Acquisition sampling rate in Hz")
    ecg: List[List[float]] = Field(..., description="ECG waveform as [leads][samples]")
    clinical: Dict[str, Optional[float]] = Field(..., description="Clinical feature mapping matching the trained schema")
    architecture: Optional[str] = Field("active", description="active, multimodal_gated, multimodal_transformer, ecg_transformer, ecg_cnn_bilstm, clinical_mlp")


class ModalityContributions(BaseModel):
    ecg_weight: float = Field(0.5, description="Relative estimated contribution of ECG signal")
    clinical_weight: float = Field(0.5, description="Relative estimated contribution of clinical data")


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    architecture: str = "active"
    research_score: float = Field(..., description="Calculated experimental risk probability [0, 1]")
    decision_threshold: float
    prediction_label: str
    modality_contributions: Dict[str, float]
    latency_ms: int
    disclaimer: str


class ExplainRequest(PredictRequest):
    steps: int = Field(10, ge=2, le=50, description="Integrated Gradients interpolation steps")


class ClinicalAttributionItem(BaseModel):
    feature: str
    normalized_value: float
    attribution_score: float
    direction: str


class LeadRankingItem(BaseModel):
    lead: str
    weight: float


class ExplainResponse(BaseModel):
    request_id: str
    research_score: float
    lead_ranking: List[LeadRankingItem]
    clinical_attributions: List[ClinicalAttributionItem]
    saliency_preview: List[List[float]] = Field(..., description="Subsampled or full saliency matrix")
    method: str
    disclaimer: str


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str


class RegisterUserRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)
    role: str = Field("researcher", description="admin, researcher, reviewer, api_client")


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    is_active: bool


class ModelMetadataResponse(BaseModel):
    model_version: str
    model_configured: bool
    artifact_path: Optional[str] = None
    artifact_sha256: Optional[str] = None
    threshold: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    model_card: Dict[str, Any] = Field(default_factory=dict)
    status: str
    disclaimer: str


class LoadModelRequest(BaseModel):
    artifact_path: str
