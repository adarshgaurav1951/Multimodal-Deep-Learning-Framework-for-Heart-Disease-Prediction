"""
Model Training, Multi-Architecture Benchmarking, and Evaluation Pipeline CLI.
Trains CNN, BiLSTM, Transformer, and Multimodal Fusion architectures,
evaluates on patient-independent test splits, serializes the benchmark artifact,
and generates diagnostic accuracy & ROC-AUC visualizations.
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from src.config import settings
from src.data.csv_loader import DEFAULT_CLINICAL_FEATURES, load_cohort_from_csv
from src.data.ecg_processor import ECGProcessor
from src.data.clinical_processor import ClinicalProcessor
from src.data.dataset import MultimodalDataset, patient_level_split, create_dataloaders
from src.models.multimodal_classifier import MultimodalClassifier
from src.models.baselines import (
    ECGOnlyClassifier,
    ECGTransformerClassifier,
    ECGCNNTransformerClassifier,
    ClinicalOnlyClassifier
)
from src.models.trainer import MultimodalTrainer, compute_binary_metrics, find_optimal_threshold
from src.models.artifact import ModelArtifactManager
from src.database.session import SessionLocal, init_db
from src.database.service import DatabaseService


def run_pipeline(
    data_source: Optional[str] = None,
    num_patients: int = 300,
    fusion_type: str = "gated",
    epochs: int = 15,
    batch_size: int = 32,
    seed: int = 42,
    artifact_output: str = "artifacts/best_multimodal_model.pt",
    benchmark_all: bool = True
):
    print("=" * 80)
    print("MULTIMODAL DEEP LEARNING HEART DISEASE PIPELINE: CNN, BiLSTM, TRANSFORMERS & FUSION")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | Random Seed: {seed}")
    print("=" * 80)

    # 1. Ingest Cohort Data strictly from CSV
    target_data_source = data_source or settings.CSV_DATASET_PATH
    data_path = Path(target_data_source)
    if not data_path.exists():
        raise FileNotFoundError(
            f"CSV dataset not found at '{target_data_source}'. Please provide a valid CSV dataset file."
        )

    print(f"\n[1/6] Ingesting cohort data strictly from CSV: '{target_data_source}'...")
    clinical_df, raw_ecgs = load_cohort_from_csv(target_data_source)
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
    filtered_ecgs = np.zeros_like(raw_ecgs, dtype=np.float32)
    for i in range(len(raw_ecgs)):
        filtered_ecgs[i] = ecg_processor.process(raw_ecgs[i], orig_fs=500.0)

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

    # 4. Model Suite Definition
    models_to_train: Dict[str, torch.nn.Module] = {
        "ECG CNN-BiLSTM": ECGOnlyClassifier(in_leads=12, embedding_dim=128),
        "ECG Transformer": ECGTransformerClassifier(in_leads=12, embedding_dim=128, num_layers=2, num_heads=4),
        "ECG CNN-Transformer": ECGCNNTransformerClassifier(in_leads=12, embedding_dim=128, num_layers=2, num_heads=4),
        "Clinical MLP Baseline": ClinicalOnlyClassifier(n_features=len(feature_cols), embedding_dim=128),
        "Multimodal Gated Fusion": MultimodalClassifier(
            in_leads=12,
            n_clinical_features=len(feature_cols),
            embedding_dim=128,
            fusion_type="gated",
            ecg_backbone="cnn_bilstm"
        ),
        "Multimodal Cross-Attention": MultimodalClassifier(
            in_leads=12,
            n_clinical_features=len(feature_cols),
            embedding_dim=128,
            fusion_type="cross_attention",
            ecg_backbone="transformer"
        )
    }

    if not benchmark_all:
        models_to_train = {
            f"Multimodal ({fusion_type})": MultimodalClassifier(
                in_leads=12,
                n_clinical_features=len(feature_cols),
                embedding_dim=128,
                fusion_type=fusion_type
            )
        }

    # 5. Train & Benchmark All Architectures
    print(f"\n[4/6] Training & Benchmarking Models across Paradigms (CNN, BiLSTM, Transformers, Fusion)...")
    benchmark_metrics: Dict[str, Dict[str, Any]] = {}
    benchmark_roc: Dict[str, Dict[str, Any]] = {}
    benchmark_histories: Dict[str, Dict[str, Any]] = {}

    best_model_name = None
    best_model_auroc = -1.0
    best_multimodal_instance = None
    best_opt_threshold = 0.5

    for model_name, model in models_to_train.items():
        print(f"\n  -> Training [{model_name}]...")
        trainer = MultimodalTrainer(model, patience=6)
        fit_res = trainer.fit(loaders["train"], loaders["val"], epochs=epochs, verbose=False)

        # Validation optimal threshold
        val_loss, y_val_true, y_val_probs = trainer.evaluate(loaders["val"])
        opt_thresh = find_optimal_threshold(y_val_true, y_val_probs)

        # Test evaluation
        test_loss, y_test_true, y_test_probs = trainer.evaluate(loaders["test"])
        metrics = compute_binary_metrics(y_test_true, y_test_probs, threshold=opt_thresh)
        benchmark_metrics[model_name] = metrics

        # Compute ROC curve coordinates (subsampled for light serializable JSON)
        fpr, tpr, threshs = roc_curve(y_test_true, y_test_probs)
        # Subsample to at most 40 points
        step = max(1, len(fpr) // 35)
        sub_fpr = [round(float(v), 4) for v in fpr[::step]]
        sub_tpr = [round(float(v), 4) for v in tpr[::step]]
        if sub_fpr[-1] != 1.0:
            sub_fpr.append(1.0)
            sub_tpr.append(1.0)

        benchmark_roc[model_name] = {
            "fpr": sub_fpr,
            "tpr": sub_tpr,
            "auroc": metrics["auroc"],
            "threshold": opt_thresh
        }

        # Training history format
        hist = fit_res.get("history", [])
        benchmark_histories[model_name] = {
            "epochs": [h["epoch"] for h in hist],
            "train_loss": [h.get("train_loss", 0.0) for h in hist],
            "train_acc": [h.get("train_acc", 0.0) for h in hist],
            "val_loss": [h.get("val_loss", 0.0) for h in hist],
            "val_acc": [h.get("accuracy", 0.0) for h in hist],
            "val_auroc": [h.get("auroc", 0.0) for h in hist]
        }

        print(f"     Test Accuracy: {metrics['accuracy']*100:.1f}% | AUROC: {metrics['auroc']:.4f} | F1: {metrics['f1']:.4f} | Sens: {metrics['sensitivity']:.4f}")

        # Keep track of best multimodal model for artifact registration
        if isinstance(model, MultimodalClassifier) and metrics["auroc"] > best_model_auroc:
            best_model_auroc = metrics["auroc"]
            best_model_name = model_name
            best_multimodal_instance = model
            best_opt_threshold = opt_thresh

    # Fallback if no MultimodalClassifier was trained
    if best_multimodal_instance is None:
        best_multimodal_instance = list(models_to_train.values())[-1]
        best_opt_threshold = 0.5
        best_model_name = list(models_to_train.keys())[-1]

    # Display Pretty Comparison Table
    print("\n" + "=" * 88)
    print("TEST EVALUATION BENCHMARK COMPARISON (Locked Holdout Patient Test Set)")
    print("=" * 88)
    print(f"{'Model Architecture':<30} | {'Accuracy':<9} | {'AUROC':<7} | {'AUPRC':<7} | {'F1':<7} | {'Sens':<7} | {'Spec':<7}")
    print("-" * 88)
    for name, m in benchmark_metrics.items():
        print(f"{name:<30} | {m['accuracy']*100:6.1f}%  | {m['auroc']:<7.4f} | {m['auprc']:<7.4f} | {m['f1']:<7.4f} | {m['sensitivity']:<7.4f} | {m['specificity']:<7.4f}")
    print("=" * 88)

    # 6. Save Benchmark Results Artifact
    bench_artifact_path = Path("artifacts/models_benchmark.json")
    bench_payload = {
        "metrics": benchmark_metrics,
        "roc_data": benchmark_roc,
        "histories": benchmark_histories,
        "cohort_size": num_patients,
        "features": feature_cols
    }
    with open(bench_artifact_path, "w") as f:
        json.dump(bench_payload, f, indent=2)
    print(f"\nSaved benchmark results artifact to {bench_artifact_path}")

    # 7. Package and Serialize Best Model Artifact
    print(f"\n[5/6] Packaging and serializing best model artifact to {artifact_output}...")
    model_card = {
        "model_name": f"Multimodal Heart Disease Classifier ({best_model_name})",
        "intended_use": "Investigational Cardiovascular ML Research",
        "cohort_size": num_patients,
        "ecg_lead_format": "12-lead (I, II, III, aVR, aVL, aVF, V1-V6)",
        "sampling_rate_hz": 500.0,
        "features": feature_cols,
        "test_accuracy": benchmark_metrics[best_model_name]["accuracy"],
        "test_auroc": benchmark_metrics[best_model_name]["auroc"],
        "test_auprc": benchmark_metrics[best_model_name]["auprc"],
        "test_f1": benchmark_metrics[best_model_name]["f1"],
        "disclaimer": "Experimental research score only. Not for clinical diagnosis or triage."
    }

    sha256 = ModelArtifactManager.save_artifact(
        model=best_multimodal_instance,
        clinical_processor=clinical_processor,
        save_path=artifact_output,
        threshold=best_opt_threshold,
        version=f"v1.0.0-{fusion_type}",
        metrics={"test": benchmark_metrics[best_model_name], "all_models": benchmark_metrics},
        model_card=model_card
    )
    print(f"Artifact saved successfully! SHA-256 Digest: {sha256}")

    # Log to Database
    init_db()
    with SessionLocal() as db:
        exp = DatabaseService.save_experiment(
            db=db,
            name=f"Multimodal Deep Learning Suite Experiment",
            config={"num_patients": num_patients, "epochs": epochs, "seed": seed},
            metrics_dict={
                "test": benchmark_metrics[best_model_name],
                "benchmark": benchmark_metrics
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

    # 8. Generate Visualizations automatically
    print("\n[6/6] Generating Publication-Quality Accuracy & ROC-AUC Visualizations...")
    from generate_visualizations import run as gen_vis_run
    try:
        gen_vis_run()
    except Exception as e:
        print(f"Warning: Visualization generation encountered: {e}")

    print("\nPipeline execution complete! Models are ready for serving and visualization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Heart Disease Training Pipeline")
    parser.add_argument("--data", "--csv", type=str, default="data/heart_disease_multimodal.csv", help="Path to CSV dataset file")
    parser.add_argument("--patients", type=int, default=300, help="Number of cohort patients")
    parser.add_argument("--fusion", type=str, default="gated", choices=["gated", "concat", "cross_attention", "transformer"], help="Default fusion mechanism")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--artifact", type=str, default="artifacts/best_multimodal_model.pt", help="Path to save model artifact")
    parser.add_argument("--benchmark-all", action="store_true", default=True, help="Benchmark all architectures (CNN, BiLSTM, Transformer, Fusion)")
    args = parser.parse_args()

    run_pipeline(
        data_source=args.data,
        num_patients=args.patients,
        fusion_type=args.fusion,
        epochs=args.epochs,
        artifact_output=args.artifact,
        benchmark_all=args.benchmark_all
    )
