"""
Unit tests for Neural Network Encoders, Fusion Heads, and Multimodal Classifiers.
"""
import torch
import numpy as np
from src.models.ecg_encoder import ECGEncoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.fusion import ConcatFusion, GatedFusion, CrossAttentionFusion
from src.models.multimodal_classifier import MultimodalClassifier
from src.models.explainability import compute_ecg_saliency, compute_clinical_attributions


def test_ecg_encoder_forward():
    encoder = ECGEncoder(in_leads=12, embedding_dim=128)
    dummy_ecg = torch.randn(4, 12, 1000)
    embedding = encoder(dummy_ecg)
    assert embedding.shape == (4, 128)


def test_clinical_encoder_forward():
    encoder = ClinicalEncoder(n_features=10, embedding_dim=128)
    dummy_clin = torch.randn(4, 10)
    embedding = encoder(dummy_clin)
    assert embedding.shape == (4, 128)


def test_gated_fusion_forward():
    fusion = GatedFusion(ecg_dim=128, clinical_dim=128, out_dim=128)
    z_ecg = torch.randn(4, 128)
    z_clin = torch.randn(4, 128)
    fused, weights = fusion(z_ecg, z_clin)
    assert fused.shape == (4, 128)
    assert "ecg_weight" in weights
    assert "clinical_weight" in weights


def test_multimodal_classifier_forward():
    model = MultimodalClassifier(in_leads=12, n_clinical_features=10, embedding_dim=128, fusion_type="gated")
    ecg = torch.randn(2, 12, 1000)
    clin = torch.randn(2, 10)
    
    logits, details = model(ecg, clin, return_details=True)
    assert logits.shape == (2,)
    assert "z_ecg" in details
    assert "z_clinical" in details

    probs = model.predict_proba(ecg, clin)
    assert probs.shape == (2,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_explainability_computations():
    model = MultimodalClassifier(in_leads=12, n_clinical_features=6, embedding_dim=64, fusion_type="gated")
    ecg = torch.randn(1, 12, 500)
    clin = torch.randn(1, 6)

    saliency = compute_ecg_saliency(model, ecg, clin, steps=5)
    assert "lead_ranking" in saliency
    assert len(saliency["lead_ranking"]) == 12

    feature_names = ["age", "sex", "bp", "chol", "hr", "angina"]
    attributions = compute_clinical_attributions(model, ecg, clin, feature_names=feature_names)
    assert len(attributions) == 6
    assert "attribution_score" in attributions[0]
