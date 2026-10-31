"""
Explainability Subsystem.
Computes:
1. ECG Saliency Maps via Integrated Gradients / Input Gradients.
2. Clinical Tabular Feature Attributions.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.data.csv_loader import STANDARD_12_LEADS, DEFAULT_CLINICAL_FEATURES


def compute_ecg_saliency(
    model: nn.Module,
    ecg_tensor: torch.Tensor,
    clinical_tensor: torch.Tensor,
    steps: int = 10,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Computes Integrated Gradients attribution for ECG input waveform.
    ecg_tensor: [1, leads, samples] or [leads, samples]
    clinical_tensor: [1, n_features] or [n_features]
    """
    model.eval()
    if ecg_tensor.ndim == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)
    if clinical_tensor.ndim == 1:
        clinical_tensor = clinical_tensor.unsqueeze(0)

    ecg_tensor = ecg_tensor.to(device).float()
    clinical_tensor = clinical_tensor.to(device).float()

    # Baseline for ECG: zero waveform
    baseline_ecg = torch.zeros_like(ecg_tensor)
    
    # Path integration approximation
    grads_sum = torch.zeros_like(ecg_tensor)
    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = baseline_ecg + alpha * (ecg_tensor - baseline_ecg)
        interpolated.requires_grad_(True)
        
        logits = model(interpolated, clinical_tensor)
        score = torch.sigmoid(logits)
        score.backward(retain_graph=True)
        
        if interpolated.grad is not None:
            grads_sum += interpolated.grad.detach()
            interpolated.grad.zero_()

    # Integrated Gradients = (Input - Baseline) * Average Gradient
    ig_attribution = (ecg_tensor - baseline_ecg) * (grads_sum / steps)
    saliency = ig_attribution.squeeze(0).abs().cpu().numpy()  # [leads, samples]

    # Normalize saliency to [0, 1] range for visual display
    max_val = np.max(saliency) if np.max(saliency) > 0 else 1.0
    norm_saliency = (saliency / max_val).tolist()

    # Compute lead-level importance
    lead_importance = np.mean(saliency, axis=1)  # [leads]
    lead_sum = np.sum(lead_importance) if np.sum(lead_importance) > 0 else 1.0
    lead_ranking = [
        {"lead": STANDARD_12_LEADS[i] if i < len(STANDARD_12_LEADS) else f"Lead_{i}", "weight": round(float(lead_importance[i] / lead_sum), 4)}
        for i in range(len(lead_importance))
    ]
    lead_ranking.sort(key=lambda x: x["weight"], reverse=True)

    return {
        "lead_ranking": lead_ranking,
        "saliency_matrix": norm_saliency,
        "method": "Integrated Gradients (n_steps=10)"
    }


def compute_clinical_attributions(
    model: nn.Module,
    ecg_tensor: torch.Tensor,
    clinical_tensor: torch.Tensor,
    feature_names: List[str] = None,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Computes gradient-based feature attribution for tabular clinical inputs.
    """
    model.eval()
    if ecg_tensor.ndim == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)
    if clinical_tensor.ndim == 1:
        clinical_tensor = clinical_tensor.unsqueeze(0)

    ecg_tensor = ecg_tensor.to(device).float()
    clinical_tensor = clinical_tensor.to(device).float()
    clinical_tensor.requires_grad_(True)

    logits = model(ecg_tensor, clinical_tensor)
    prob = torch.sigmoid(logits)
    prob.backward()

    grads = clinical_tensor.grad.squeeze(0).cpu().numpy()
    values = clinical_tensor.squeeze(0).detach().cpu().numpy()

    # Attribution = feature_value * gradient
    attribution = np.abs(values * grads)
    total_attr = np.sum(attribution) if np.sum(attribution) > 0 else 1.0

    features = feature_names or DEFAULT_CLINICAL_FEATURES
    results = []
    for i, name in enumerate(features[:len(attribution)]):
        raw_val = float(values[i])
        attr_pct = float(attribution[i] / total_attr)
        direction = "increases_risk" if grads[i] > 0 else "decreases_risk"
        results.append({
            "feature": name,
            "normalized_value": round(raw_val, 3),
            "attribution_score": round(attr_pct, 4),
            "direction": direction
        })

    results.sort(key=lambda x: x["attribution_score"], reverse=True)
    return results
