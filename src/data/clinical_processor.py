"""
Clinical Tabular Data Processing Pipeline.
Implements missingness-aware handling, training-only median imputation,
and feature scaling with serializable state.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from src.data.contracts import DEFAULT_CLINICAL_FEATURES, CLINICAL_FEATURE_SCHEMA


class ClinicalProcessor:
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        add_missing_indicators: bool = False
    ):
        self.feature_names = feature_names or list(DEFAULT_CLINICAL_FEATURES)
        self.add_missing_indicators = add_missing_indicators
        
        # Fitted parameters (must only be fitted on training data)
        self.impute_values_: Dict[str, float] = {}
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.is_fitted: bool = False

    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Validate single record features against known schema.
        Replaces non-finite values with None (to be imputed).
        """
        cleaned: Dict[str, Optional[float]] = {}
        for feat in self.feature_names:
            val = features.get(feat, None)
            if val is not None:
                try:
                    fval = float(val)
                    if np.isfinite(fval):
                        # Bounds check warnings or clamping if appropriate
                        spec = CLINICAL_FEATURE_SCHEMA.get(feat, {})
                        min_v = spec.get("min", -np.inf)
                        max_v = spec.get("max", np.inf)
                        cleaned[feat] = max(min_v, min(max_v, fval))
                    else:
                        cleaned[feat] = None
                except (ValueError, TypeError):
                    cleaned[feat] = None
            else:
                cleaned[feat] = None
        return cleaned

    def fit(self, data: Union[pd.DataFrame, List[Dict[str, Optional[float]]]]) -> "ClinicalProcessor":
        """
        Fit median imputation and z-score scaling on TRAINING data only.
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Compute median imputation values and scaling statistics for each feature
        for feat in self.feature_names:
            if feat in df.columns:
                series = pd.to_numeric(df[feat], errors='coerce')
                valid_vals = series.dropna()
                if len(valid_vals) > 0:
                    med = float(valid_vals.median())
                    mean = float(valid_vals.mean())
                    std = float(valid_vals.std(ddof=0))
                    if std < 1e-6:
                        std = 1.0
                else:
                    med = 0.0
                    mean = 0.0
                    std = 1.0
            else:
                med = 0.0
                mean = 0.0
                std = 1.0

            self.impute_values_[feat] = med
            self.means_[feat] = mean
            self.stds_[feat] = std

        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, List[Dict[str, Optional[float]]], Dict[str, Optional[float]]]) -> np.ndarray:
        """
        Transform clinical features into normalized numerical numpy array [N, num_features].
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalProcessor must be fitted before transform")

        is_single = False
        if isinstance(data, dict):
            records = [self.validate_features(data)]
            is_single = True
        elif isinstance(data, list):
            records = [self.validate_features(r) for r in data]
        elif isinstance(data, pd.DataFrame):
            records = [self.validate_features(row) for row in data.to_dict(orient="records")]
        else:
            raise ValueError(f"Unsupported data type for transform: {type(data)}")

        rows = []
        for r in records:
            row_features = []
            for feat in self.feature_names:
                raw_val = r.get(feat, None)
                is_missing = 1.0 if raw_val is None else 0.0
                val = self.impute_values_[feat] if raw_val is None else float(raw_val)
                # Standardize
                norm_val = (val - self.means_[feat]) / self.stds_[feat]
                row_features.append(norm_val)
                if self.add_missing_indicators:
                    row_features.append(is_missing)
            rows.append(row_features)

        arr = np.array(rows, dtype=np.float32)
        if is_single:
            return arr[0]
        return arr

    def fit_transform(self, data: Union[pd.DataFrame, List[Dict[str, Optional[float]]]]) -> np.ndarray:
        return self.fit(data).transform(data)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize processor state for model artifact."""
        return {
            "feature_names": self.feature_names,
            "add_missing_indicators": self.add_missing_indicators,
            "impute_values": self.impute_values_,
            "means": self.means_,
            "stds": self.stds_,
            "is_fitted": self.is_fitted
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "ClinicalProcessor":
        """Load fitted parameters into processor."""
        self.feature_names = state["feature_names"]
        self.add_missing_indicators = state.get("add_missing_indicators", False)
        self.impute_values_ = {k: float(v) for k, v in state["impute_values"].items()}
        self.means_ = {k: float(v) for k, v in state["means"].items()}
        self.stds_ = {k: float(v) for k, v in state["stds"].items()}
        self.is_fitted = state.get("is_fitted", True)
        return self
