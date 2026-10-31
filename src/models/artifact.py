"""
Model Artifact Serialization, Integrity Hashing, and Packaging.
Manages .pt artifact bundles containing model weights, architecture configs,
preprocessing statistics, decision thresholds, and model cards.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import os
import hashlib
from pathlib import Path
import torch
from src.models.multimodal_classifier import MultimodalClassifier
from src.data.clinical_processor import ClinicalProcessor
from src.data.ecg_processor import ECGProcessor


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ModelArtifactManager:
    @staticmethod
    def save_artifact(
        model: MultimodalClassifier,
        clinical_processor: ClinicalProcessor,
        save_path: str,
        threshold: float = 0.5,
        version: str = "v1.0.0",
        metrics: Optional[Dict[str, Any]] = None,
        model_card: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Package model state, configs, preprocessing parameters, and metadata into a single .pt artifact.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        artifact = {
            "version": version,
            "architecture": {
                "in_leads": model.in_leads,
                "n_clinical_features": model.n_clinical_features,
                "embedding_dim": model.embedding_dim,
                "fusion_type": model.fusion_type,
                "ecg_backbone": getattr(model, "ecg_backbone", "cnn_bilstm")
            },
            "state_dict": model.state_dict(),
            "clinical_processor_state": clinical_processor.state_dict(),
            "threshold": threshold,
            "metrics": metrics or {},
            "model_card": model_card or {
                "name": "Multimodal Heart Disease Classifier",
                "version": version,
                "intended_use": "Investigational Research & Educational Prototype Only",
                "disclaimer": "Not a diagnostic device. Experimental score only."
            }
        }

        torch.save(artifact, save_path)
        sha256_hash = compute_sha256(save_path)
        return sha256_hash

    @staticmethod
    def load_artifact(
        artifact_path: str,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Safely load model artifact and instantiate preconfigured model and processors.
        """
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

        # Compute hash
        sha256 = compute_sha256(artifact_path)

        # Load bundle (weights_only=False because artifact dict contains processor states)
        artifact = torch.load(artifact_path, map_location=device, weights_only=False)

        arch = artifact["architecture"]
        model = MultimodalClassifier(
            in_leads=arch["in_leads"],
            n_clinical_features=arch["n_clinical_features"],
            embedding_dim=arch["embedding_dim"],
            fusion_type=arch["fusion_type"],
            ecg_backbone=arch.get("ecg_backbone", "cnn_bilstm")
        )
        model.load_state_dict(artifact["state_dict"])
        model.to(device)
        model.eval()

        # Reconstruct clinical processor
        clinical_proc = ClinicalProcessor()
        clinical_proc.load_state_dict(artifact["clinical_processor_state"])

        # Instantiate ECG processor
        ecg_proc = ECGProcessor(target_fs=500.0, nominal_samples=2500)

        return {
            "model": model,
            "clinical_processor": clinical_proc,
            "ecg_processor": ecg_proc,
            "threshold": artifact.get("threshold", 0.5),
            "version": artifact.get("version", "unknown"),
            "metrics": artifact.get("metrics", {}),
            "model_card": artifact.get("model_card", {}),
            "sha256": sha256
        }
