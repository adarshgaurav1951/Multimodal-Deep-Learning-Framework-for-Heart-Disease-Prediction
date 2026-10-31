"""
Model Evaluation and Diagnostic Reporting Script.
Loads registered model artifact, runs locked test evaluation, calibration assessment,
and explainability verification.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from src.config import settings
from src.models.artifact import ModelArtifactManager
from src.data.csv_loader import load_cohort_from_csv
from src.models.trainer import compute_binary_metrics
from src.models.explainability import compute_ecg_saliency, compute_clinical_attributions


def evaluate_artifact(
    artifact_path: str,
    data_source: Optional[str] = None
):
    print(f"Loading artifact from {artifact_path}...")
    bundle = ModelArtifactManager.load_artifact(artifact_path)
    model = bundle["model"]
    clinical_processor = bundle["clinical_processor"]
    ecg_processor = bundle["ecg_processor"]
    threshold = bundle["threshold"]

    print(f"Model version: {bundle['version']}")
    print(f"Artifact SHA-256: {bundle['sha256']}")
    print(f"Tuned decision threshold: {threshold:.3f}")

    # Load cohort data strictly from CSV
    target_source = data_source or settings.CSV_DATASET_PATH
    if not Path(target_source).exists():
        raise FileNotFoundError(f"Evaluation CSV source not found at '{target_source}'.")

    print(f"\nLoading evaluation cohort strictly from CSV source: '{target_source}'...")
    df_eval, raw_ecgs = load_cohort_from_csv(target_source)

    # Preprocess
    proc_ecgs = np.zeros_like(raw_ecgs, dtype=np.float32)
    for i in range(len(raw_ecgs)):
        proc_ecgs[i] = ecg_processor.process(raw_ecgs[i], orig_fs=500.0)

    X_clin = clinical_processor.transform(df_eval[clinical_processor.feature_names])
    y_true = df_eval["target"].to_numpy()

    # Predict
    model.eval()
    with torch.no_grad():
        ecg_t = torch.from_numpy(proc_ecgs).float()
        clin_t = torch.from_numpy(X_clin).float()
        logits = model(ecg_t, clin_t)
        probs = torch.sigmoid(logits).numpy()

    metrics = compute_binary_metrics(y_true, probs, threshold=threshold)

    print("\n" + "=" * 60)
    print("INDEPENDENT EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k.capitalize():<20}: {v}")
    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['confusion_matrix']['tp']:<5} | FP: {metrics['confusion_matrix']['fp']:<5}")
    print(f"  FN: {metrics['confusion_matrix']['fn']:<5} | TN: {metrics['confusion_matrix']['tn']:<5}")
    print("=" * 60)

    # Test explainability on 1 sample
    print("\nRunning Explainability on sample 0...")
    saliency = compute_ecg_saliency(model, ecg_t[0], clin_t[0], steps=10)
    print("Top 3 ECG leads by saliency:")
    for item in saliency["lead_ranking"][:3]:
        print(f"  - {item['lead']}: weight={item['weight']}")

    attributions = compute_clinical_attributions(model, ecg_t[0], clin_t[0], clinical_processor.feature_names)
    print("Top 3 Clinical features by attribution:")
    for item in attributions[:3]:
        print(f"  - {item['feature']}: score={item['attribution_score']} ({item['direction']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Heart Disease Model Artifact")
    parser.add_argument("--artifact", type=str, default="artifacts/best_multimodal_model.pt", help="Artifact path")
    parser.add_argument("--data", "--csv", type=str, default="data/heart_disease_multimodal.csv", help="Path to CSV dataset file (.csv) or directory containing CSV files")
    parser.add_argument("--patients", type=int, default=100, help="Number of evaluation patients (if generating new CSV)")
    args = parser.parse_args()

    evaluate_artifact(args.artifact, data_source=args.data)
