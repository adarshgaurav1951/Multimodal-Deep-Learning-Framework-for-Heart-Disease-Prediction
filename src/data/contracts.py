"""
Data contracts, schema definitions, and validation structures.
Ensures consistency between raw datasets, training pipelines, and serving endpoints.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator
import numpy as np

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

CLINICAL_FEATURE_SCHEMA = {
    "age": {"type": "float", "unit": "years", "min": 1.0, "max": 120.0, "description": "Patient age in years"},
    "sex": {"type": "float", "unit": "binary", "min": 0.0, "max": 1.0, "description": "1 = Male, 0 = Female"},
    "resting_bp": {"type": "float", "unit": "mmHg", "min": 50.0, "max": 300.0, "description": "Resting systolic blood pressure"},
    "cholesterol": {"type": "float", "unit": "mg/dL", "min": 50.0, "max": 800.0, "description": "Serum cholesterol level"},
    "fasting_bs": {"type": "float", "unit": "binary", "min": 0.0, "max": 1.0, "description": "Fasting blood sugar > 120 mg/dl (1=true, 0=false)"},
    "resting_ecg": {"type": "float", "unit": "category", "min": 0.0, "max": 2.0, "description": "0=Normal, 1=ST-T wave abnormality, 2=LVH"},
    "max_hr": {"type": "float", "unit": "bpm", "min": 30.0, "max": 250.0, "description": "Maximum heart rate achieved"},
    "exercise_angina": {"type": "float", "unit": "binary", "min": 0.0, "max": 1.0, "description": "Exercise induced angina (1=yes, 0=no)"},
    "st_depression": {"type": "float", "unit": "mm", "min": 0.0, "max": 15.0, "description": "ST depression induced by exercise relative to rest"},
    "st_slope": {"type": "float", "unit": "category", "min": 0.0, "max": 2.0, "description": "0=Upsloping, 1=Flat, 2=Downsloping"},
    "num_major_vessels": {"type": "float", "unit": "count", "min": 0.0, "max": 4.0, "description": "Number of major vessels (0-4) colored by fluoroscopy"},
    "thal": {"type": "float", "unit": "category", "min": 0.0, "max": 3.0, "description": "0=Normal, 1=Fixed defect, 2=Reversible defect"}
}

DEFAULT_CLINICAL_FEATURES = list(CLINICAL_FEATURE_SCHEMA.keys())


class ECGSignalMetadata(BaseModel):
    record_id: str
    patient_id: Optional[str] = None
    sampling_rate_hz: float = Field(gt=0.0, le=5000.0)
    lead_names: List[str] = Field(default_factory=lambda: list(STANDARD_12_LEADS))
    units: str = "mV"
    duration_sec: Optional[float] = None


class ClinicalRecord(BaseModel):
    record_id: str
    patient_id: Optional[str] = None
    features: Dict[str, Optional[float]]
    target: Optional[int] = Field(default=None, ge=0, le=1)

    @field_validator("features")
    def validate_features_non_empty(cls, v):
        if not v:
            raise ValueError("Clinical features dictionary cannot be empty")
        return v


class MultimodalRecord(BaseModel):
    record_id: str
    patient_id: str
    sampling_rate_hz: float
    ecg: List[List[float]]
    clinical: Dict[str, Optional[float]]
    target: Optional[int] = None


class DatasetManifest(BaseModel):
    dataset_name: str
    version: str
    num_records: int
    num_patients: int
    sampling_rate_hz: float
    leads: List[str]
    clinical_features: List[str]
    split_counts: Dict[str, int]
    manifest_hash: str
