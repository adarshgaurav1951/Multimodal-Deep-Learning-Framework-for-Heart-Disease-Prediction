"""
Model Training, Validation, and Metric Evaluation Engine.
Provides reproducible training loops, early stopping, and clinical evaluation metrics.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    precision_score,
    recall_score
)


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute full cardiovascular risk evaluation metrics suite.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    y_pred = (probs >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Sensitivity / Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # ROC and PR metrics (handle edge cases with single class)
    try:
        auroc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(y_true, probs))
    except ValueError:
        auprc = float(np.mean(y_true))

    brier = float(brier_score_loss(y_true, probs))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(prec, 4),
        "npv": round(float(npv), 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
    }


def find_optimal_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J statistic (Sensitivity + Specificity - 1) on validation set."""
    best_thresh = 0.5
    best_j = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (probabilities >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thresh = t
    return float(best_thresh)


class MultimodalTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        patience: int = 7
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.patience = patience

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for ecg, clinical, target in loader:
            ecg = ecg.to(self.device).float()
            clinical = clinical.to(self.device).float()
            target = target.to(self.device).float()

            self.optimizer.zero_grad(set_to_none=True)
            # Check model type
            if hasattr(self.model, "clinical_encoder") and hasattr(self.model, "ecg_encoder"):
                logits = self.model(ecg, clinical)
            elif hasattr(self.model, "encoder") and ecg.shape[1] > 1:
                # ECG only or Clinical only
                logits = self.model(ecg) if hasattr(self.model, "head") and hasattr(self.model.encoder, "lstm") else self.model(clinical)
            else:
                logits = self.model(ecg, clinical)

            loss = self.criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = len(target)
            total_loss += loss.item() * batch_size
            count += batch_size

        return total_loss / max(1, count)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        all_probs = []
        all_targets = []

        for ecg, clinical, target in loader:
            ecg = ecg.to(self.device).float()
            clinical = clinical.to(self.device).float()
            target = target.to(self.device).float()

            if hasattr(self.model, "clinical_encoder") and hasattr(self.model, "ecg_encoder"):
                logits = self.model(ecg, clinical)
            else:
                logits = self.model(ecg, clinical)

            loss = self.criterion(logits, target)
            probs = torch.sigmoid(logits)

            batch_size = len(target)
            total_loss += loss.item() * batch_size
            count += batch_size

            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

        val_loss = total_loss / max(1, count)
        return val_loss, np.array(all_targets), np.array(all_probs)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25,
        verbose: bool = True
    ) -> Dict[str, Any]:
        best_auroc = 0.0
        best_weights = None
        best_val_metrics = {}
        no_improve_epochs = 0
        history = []

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, y_true, y_probs = self.evaluate(val_loader)
            scheduler.step()

            metrics = compute_binary_metrics(y_true, y_probs, threshold=0.5)
            metrics["train_loss"] = round(train_loss, 4)
            metrics["val_loss"] = round(val_loss, 4)
            metrics["epoch"] = epoch
            history.append(metrics)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {metrics['auroc']:.4f} | Val F1: {metrics['f1']:.4f}")

            # Early stopping check on validation AUROC
            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                best_weights = copy.deepcopy(self.model.state_dict())
                # Tune optimal threshold on validation set
                optimal_thresh = find_optimal_threshold(y_true, y_probs)
                best_val_metrics = compute_binary_metrics(y_true, y_probs, threshold=optimal_thresh)
                best_val_metrics["optimal_threshold"] = optimal_thresh
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return {
            "best_val_metrics": best_val_metrics,
            "history": history
        }
