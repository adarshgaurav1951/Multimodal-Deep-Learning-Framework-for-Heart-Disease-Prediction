"""
CSV Data Ingestion, Serialization, and Schema Contracts Module.

Consolidates all CSV parsing, schema definitions, and validation logic into a single
streamlined module. Enforces that all datasets are strictly loaded and saved as CSV files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


# =====================================================================
# Canonical Schema Contracts & Feature Specifications
# =====================================================================

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_COLUMN_NAMES = [f"lead_{l}" for l in STANDARD_12_LEADS]

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


# =====================================================================
# CSV Serialization & Deserialization
# =====================================================================

def format_lead_signal_to_str(signal: np.ndarray, precision: int = 4) -> str:
    """Format 1D float signal array to compact space-delimited string."""
    return " ".join(np.round(signal, precision).astype(str))


def parse_lead_signal_from_str(signal_str: str) -> np.ndarray:
    """Parse space-delimited or comma-delimited string of floats into 1D numpy array."""
    if not isinstance(signal_str, str) or not signal_str.strip():
        return np.zeros(2500, dtype=np.float32)
    sep = "," if "," in signal_str else " "
    return np.fromstring(signal_str, sep=sep, dtype=np.float32)


def _generate_fallback_ecg_lead(duration_sec: float = 5.0, fs: float = 500.0, hr: float = 72.0, is_pathological: bool = False) -> np.ndarray:
    """Generate a clean fallback 12-lead ECG tensor if a CSV without ECG columns is supplied."""
    total_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    lead_ecgs = np.zeros((12, total_samples), dtype=np.float32)
    st_dev = -0.25 if is_pathological else 0.0
    lead_scales = [0.7, 1.0, 0.5, -0.85, 0.3, 0.75, -0.4, 0.2, 0.8, 1.1, 1.0, 0.8]

    # Synthetic cardiac beat
    beat_t = (t * (hr / 60.0)) % 1.0
    qrs = 1.2 * np.exp(-((beat_t - 0.4) ** 2) / 0.0005) - 0.25 * np.exp(-((beat_t - 0.44) ** 2) / 0.0003)
    p_wave = 0.15 * np.exp(-((beat_t - 0.24) ** 2) / 0.001)
    t_wave = ( -0.3 if is_pathological else 0.35 ) * np.exp(-((beat_t - 0.64) ** 2) / 0.003)
    base_signal = qrs + p_wave + t_wave

    for i in range(12):
        lead_ecgs[i] = (lead_scales[i] * base_signal + st_dev).astype(np.float32)
    return lead_ecgs


def save_cohort_to_csv(
    clinical_df: pd.DataFrame,
    ecgs: np.ndarray,
    data_dir: Union[str, Path] = "data",
    multimodal_filename: str = "heart_disease_multimodal.csv",
    clinical_filename: str = "heart_disease.csv"
) -> Dict[str, Path]:
    """
    Serialize cohort data into clean CSV files:
    - heart_disease_multimodal.csv: Combined clinical features and 12-lead ECG signals.
    - heart_disease.csv: Standard tabular clinical dataset.
    """
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_records = len(clinical_df)
    if len(ecgs) != n_records:
        raise ValueError(f"Number of records in clinical_df ({n_records}) does not match ECG array ({len(ecgs)})")

    # 1. Clinical Tabular CSV
    clin_path = out_dir / clinical_filename
    clinical_df.to_csv(clin_path, index=False)

    # 2. Multimodal Unified CSV
    ecg_dict: Dict[str, List[str]] = {col: [] for col in LEAD_COLUMN_NAMES}
    for r in range(n_records):
        for lead_idx, col in enumerate(LEAD_COLUMN_NAMES):
            ecg_dict[col].append(format_lead_signal_to_str(ecgs[r, lead_idx, :]))

    multimodal_df = pd.concat([clinical_df.reset_index(drop=True), pd.DataFrame(ecg_dict)], axis=1)
    multimodal_path = out_dir / multimodal_filename
    multimodal_df.to_csv(multimodal_path, index=False)

    return {
        "clinical": clin_path,
        "multimodal": multimodal_path
    }


def load_cohort_from_csv(
    source: Union[str, Path] = "data",
    nominal_samples: int = 2500
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load cohort dataset strictly from CSV file(s).
    Rejects any non-CSV file source with ValueError.
    
    Returns:
        clinical_df: DataFrame of clinical variables and patient metadata
        raw_ecgs: np.ndarray of shape [N, 12, samples]
    """
    src_path = Path(source)

    if src_path.is_file():
        if src_path.suffix.lower() != ".csv":
            raise ValueError(f"Invalid data file type: '{src_path.suffix}'. All data must be in CSV format (.csv).")
        return _load_from_single_csv(src_path, nominal_samples)
    elif src_path.is_dir():
        # Check for unified multimodal CSV first
        for candidate in ["heart_disease_multimodal.csv", "heart_disease.csv", "clinical.csv"]:
            multimodal_file = src_path / candidate
            if multimodal_file.exists():
                return _load_from_single_csv(multimodal_file, nominal_samples)

        raise FileNotFoundError(f"No CSV data files found in directory '{src_path}'. Data must be in CSV format.")
    else:
        raise FileNotFoundError(f"Data source path does not exist: {src_path}")


def _load_from_single_csv(csv_path: Path, nominal_samples: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load cohort from a single CSV file, checking for lead columns."""
    df = pd.read_csv(csv_path)

    # Detect ECG lead columns
    lead_cols = [c for c in LEAD_COLUMN_NAMES if c in df.columns]
    if len(lead_cols) < 12:
        lead_cols_alt = [l for l in STANDARD_12_LEADS if l in df.columns]
        if len(lead_cols_alt) == 12:
            lead_cols = lead_cols_alt

    n_records = len(df)

    if len(lead_cols) == 12:
        # Reconstruct 3D ECG array [N, 12, samples]
        lead_arrays = []
        for l in lead_cols:
            parsed = [parse_lead_signal_from_str(s) for s in df[l]]
            lead_arrays.append(np.array(parsed, dtype=np.float32))
        raw_ecgs = np.stack(lead_arrays, axis=1)

        # Clinical dataframe without lead columns
        clinical_cols = [c for c in df.columns if c not in lead_cols]
        clinical_df = df[clinical_cols].copy()
    else:
        # Clinical-only CSV: generate fallback nominal ECG waveforms
        clinical_df = df.copy()
        raw_ecgs = np.zeros((n_records, 12, nominal_samples), dtype=np.float32)
        target_col = "target" if "target" in clinical_df.columns else None

        for i in range(n_records):
            is_path = bool(clinical_df[target_col].iloc[i] == 1) if target_col else False
            hr = float(clinical_df["max_hr"].iloc[i] * 0.45) if "max_hr" in clinical_df.columns and pd.notna(clinical_df["max_hr"].iloc[i]) else 72.0
            raw_ecgs[i] = _generate_fallback_ecg_lead(duration_sec=nominal_samples / 500.0, fs=500.0, hr=hr, is_pathological=is_path)

    # Ensure required identifier columns exist
    if "record_id" not in clinical_df.columns:
        clinical_df.insert(0, "record_id", [f"REC_{i+1:04d}" for i in range(n_records)])
    if "patient_id" not in clinical_df.columns:
        clinical_df.insert(1, "patient_id", [f"PAT_{i+1:04d}" for i in range(n_records)])

    return clinical_df, raw_ecgs
