"""
Model Training and Experimentation Pipeline CLI.
Trains unimodal baselines and multimodal fusion models, evaluates on patient-independent test split,
and registers the best artifact.
"""
from __future__ import annotations

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd

from src.config import settings
from src.data.synthetic_generator import generate_synthetic_multimodal_cohort
from src.data.contracts import DEFAULT_CLINICAL_FEATURES
from src.data.ecg_processor import ECGProcessor
from src.data.clinical_processor import ClinicalProcessor
from src.data.dataset import MultimodalDataset, patient_level_split, create_dataloaders
from src.models.multimodal_classifier import MultimodalClassifier
from src.models.baselines import ECGOnlyClassifier, ClinicalOnlyClassifier
from src.models.trainer import MultimodalTrainer, compute_binary_metrics, find_optimal_threshold
from src.models.artifact import ModelArtifactManager
from src.database.session import SessionLocal, init_db
from src.database.service import DatabaseService


def run_pipeline(
    num_patients: int = 300,
    fusion_type: str = "gated",
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    artifact_output: str = "artifacts/best_multimodal_model.pt"
):
    print("=" * 70)
    print("MULTIMODAL DEEP LEARNING HEART DISEASE PIPELINE")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | Random Seed: {seed}")
    print("=" * 70)

    # 1. Ingest / Generate Cohort
    print(f"\n[1/6] Ingesting cohort data ({num_patients} patients)...")
    clinical_df, raw_ecgs = generate_synthetic_multimodal_cohort(num_patients=num_patients, seed=seed)
    print(f"Total Records: {len(clinical_df)} across {clinical_df['patient_id'].nunique()} unique patients.")
    print(f"Label distribution: {clinical_df['target'].value_counts().to_dict()}")

    # 2. Leakage-Free Patient-Level Split
    print("\n[2/6] Partitioning dataset with strict GroupShuffleSplit by patient_id...")
    train_idx, val_idx, test_idx = patient_level_split(
        clinical_df,
        patient_col="patient_id",
        target_col="target",
        val_size=0.15,
        test_size=0.15,
        seed=seed
    )
    print(f"Split counts -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # 3. Preprocessing (Fitted on Training Split Only)
    print("\n[3/6] Preprocessing ECG signals & clinical features (fit on TRAIN only)...")
    ecg_processor = ECGProcessor(target_fs=500.0, nominal_samples=2500)
    # Batch filter ECGs
    filtered_ecgs = np.zeros_like(raw_ecgs, dtype=np.float32)
    for i in range(len(raw_ecgs)):
        filtered_ecgs[i] = ecg_processor.process(raw_ecgs[i], orig_fs=500.0)

    # Clinical fit on train
    feature_cols = [c for c in DEFAULT_CLINICAL_FEATURES if c in clinical_df.columns]
    clinical_processor = ClinicalProcessor(feature_names=feature_cols)
    clinical_processor.fit(clinical_df.iloc[train_idx][feature_cols])

    X_train_clin = clinical_processor.transform(clinical_df.iloc[train_idx][feature_cols])
    X_val_clin = clinical_processor.transform(clinical_df.iloc[val_idx][feature_cols])
    X_test_clin = clinical_processor.transform(clinical_df.iloc[test_idx][feature_cols])

    y_train = clinical_df.iloc[train_idx]["target"].to_numpy()
    y_val = clinical_df.iloc[val_idx]["target"].to_numpy()
    y_test = clinical_df.iloc[test_idx]["target"].to_numpy()

    train_dataset = MultimodalDataset(filtered_ecgs[train_idx], X_train_clin, y_train)
    val_dataset = MultimodalDataset(filtered_ecgs[val_idx], X_val_clin, y_val)
    test_dataset = MultimodalDataset(filtered_ecgs[test_idx], X_test_clin, y_test)

    loaders = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    # 4. Train Unimodal Baselines
    print("\n[4/6] Training Unimodal Baselines for ablation benchmarking...")
    
    # ECG-only
    print("  -> Training ECG-Only CNN-BiLSTM Baseline...")
    ecg_baseline = ECGOnlyClassifier(in_leads=12, embedding_dim=128)
    ecg_trainer = MultimodalTrainer(ecg_baseline)
    # Train using custom loop for unimodal
    for epoch in range(1, epochs + 1):
        ecg_baseline.train()
        for ecg_b, _, target_b in loaders["train"]:
            ecg_trainer.optimizer.zero_grad()
            out = ecg_baseline(ecg_b)
            loss = ecg_trainer.criterion(out, target_b)
            loss.backward()
            ecg_trainer.optimizer.step()
    ecg_baseline.eval()
    with torch.no_grad():
        ecg_test_probs = torch.sigmoid(ecg_baseline(torch.from_numpy(filtered_ecgs[test_idx]).float())).numpy()
    ecg_metrics = compute_binary_metrics(y_test, ecg_test_probs)
    print(f"     ECG Baseline Test AUROC: {ecg_metrics['auroc']} | AUPRC: {ecg_metrics['auprc']} | F1: {ecg_metrics['f1']}")

    # Clinical-only
    print("  -> Training Clinical-Only MLP Baseline...")
    clin_baseline = ClinicalOnlyClassifier(n_features=len(feature_cols), embedding_dim=128)
    clin_trainer = MultimodalTrainer(clin_baseline)
    for epoch in range(1, epochs + 1):
        clin_baseline.train()
        for _, clin_b, target_b in loaders["train"]:
            clin_trainer.optimizer.zero_grad()
            out = clin_baseline(clin_b)
            loss = clin_trainer.criterion(out, target_b)
            loss.backward()
            clin_trainer.optimizer.step()
    clin_baseline.eval()
    with torch.no_grad():
        clin_test_probs = torch.sigmoid(clin_baseline(torch.from_numpy(X_test_clin).float())).numpy()
    clin_metrics = compute_binary_metrics(y_test, clin_test_probs)
    print(f"     Clinical Baseline Test AUROC: {clin_metrics['auroc']} | AUPRC: {clin_metrics['auprc']} | F1: {clin_metrics['f1']}")

    # 5. Train Multimodal Model
    print(f"\n[5/6] Training Multimodal Deep Learning Model (Fusion: {fusion_type.upper()})...")
    multimodal_model = MultimodalClassifier(
        in_leads=12,
        n_clinical_features=len(feature_cols),
        embedding_dim=128,
        fusion_type=fusion_type
    )
    mm_trainer = MultimodalTrainer(multimodal_model, patience=8)
    fit_res = mm_trainer.fit(loaders["train"], loaders["val"], epochs=epochs, verbose=True)

    # Evaluate locked test set
    val_loss, y_val_true, y_val_probs = mm_trainer.evaluate(loaders["val"])
    opt_threshold = find_optimal_threshold(y_val_true, y_val_probs)
    print(f"Optimal decision threshold tuned on validation set: {opt_threshold:.3f}")

    test_loss, y_test_true, y_test_probs = mm_trainer.evaluate(loaders["test"])
    test_metrics = compute_binary_metrics(y_test_true, y_test_probs, threshold=opt_threshold)
    print("\n" + "=" * 70)
    print("TEST EVALUATION BENCHMARK COMPARISON (Locked Holdout Patient Test Set)")
    print("=" * 70)
    print(f"{'Model':<25} | {'AUROC':<7} | {'AUPRC':<7} | {'F1':<7} | {'Sens':<7} | {'Spec':<7} | {'Brier':<7}")
    print("-" * 70)
    print(f"{'ECG-Only Baseline':<25} | {ecg_metrics['auroc']:<7} | {ecg_metrics['auprc']:<7} | {ecg_metrics['f1']:<7} | {ecg_metrics['sensitivity']:<7} | {ecg_metrics['specificity']:<7} | {ecg_metrics['brier_score']:<7}")
    print(f"{'Clinical-Only Baseline':<25} | {clin_metrics['auroc']:<7} | {clin_metrics['auprc']:<7} | {clin_metrics['f1']:<7} | {clin_metrics['sensitivity']:<7} | {clin_metrics['specificity']:<7} | {clin_metrics['brier_score']:<7}")
    print(f"{f'Multimodal ({fusion_type})':<25} | {test_metrics['auroc']:<7} | {test_metrics['auprc']:<7} | {test_metrics['f1']:<7} | {test_metrics['sensitivity']:<7} | {test_metrics['specificity']:<7} | {test_metrics['brier_score']:<7}")
    print("=" * 70)

    # 6. Export and Register Artifact
    print(f"\n[6/6] Packaging and serializing model artifact to {artifact_output}...")
    model_card = {
        "model_name": f"Multimodal Heart Disease Classifier ({fusion_type.capitalize()} Fusion)",
        "intended_use": "Investigational Cardiovascular ML Research",
        "cohort_size": num_patients,
        "ecg_lead_format": "12-lead (I, II, III, aVR, aVL, aVF, V1-V6)",
        "sampling_rate_hz": 500.0,
        "features": feature_cols,
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "test_f1": test_metrics["f1"],
        "disclaimer": "Experimental research score only. Not for clinical diagnosis or triage."
    }

    sha256 = ModelArtifactManager.save_artifact(
        model=multimodal_model,
        clinical_processor=clinical_processor,
        save_path=artifact_output,
        threshold=opt_threshold,
        version=f"v1.0.0-{fusion_type}",
        metrics={"test": test_metrics, "ecg_baseline": ecg_metrics, "clinical_baseline": clin_metrics},
        model_card=model_card
    )
    print(f"Artifact saved successfully!")
    print(f"SHA-256 Digest: {sha256}")

    # Log to Database
    init_db()
    with SessionLocal() as db:
        exp = DatabaseService.save_experiment(
            db=db,
            name=f"Multimodal {fusion_type.capitalize()} Experiment",
            config={"num_patients": num_patients, "epochs": epochs, "fusion_type": fusion_type, "seed": seed},
            metrics_dict={
                "val": fit_res["best_val_metrics"],
                "test": test_metrics,
                "ecg_baseline": ecg_metrics,
                "clinical_baseline": clin_metrics
            }
        )
        mv = DatabaseService.register_model_version(
            db=db,
            version=f"v1.0.0-{fusion_type}",
            artifact_uri=artifact_output,
            artifact_sha256=sha256,
            model_card=model_card,
            experiment_id=exp.id
        )
        print(f"Experiment logged to database (ID: {exp.id}, ModelVersion: {mv.version})")

    print("\nPipeline execution complete! Model is ready for serving and sandbox testing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Heart Disease Training Pipeline")
    parser.add_argument("--patients", type=int, default=250, help="Number of cohort patients")
    parser.add_argument("--fusion", type=str, default="gated", choices=["gated", "concat", "cross_attention"], help="Fusion mechanism")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--artifact", type=str, default="artifacts/best_multimodal_model.pt", help="Path to save model artifact")
    args = parser.parse_args()

    run_pipeline(
        num_patients=args.patients,
        fusion_type=args.fusion,
        epochs=args.epochs,
        artifact_output=args.artifact
    )
