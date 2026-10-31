"""
Unit tests for Neural Network Encoders, Transformers, Fusion Heads, and Multimodal Classifiers.
"""
import torch
import numpy as np
from src.models.ecg_encoder import ECGEncoder
from src.models.transformer_encoder import ECGTransformerEncoder, ECGCNNTransformerEncoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.fusion import ConcatFusion, GatedFusion, CrossAttentionFusion, MultimodalTransformerFusion
from src.models.multimodal_classifier import MultimodalClassifier
from src.models.baselines import ECGOnlyClassifier, ECGTransformerClassifier, ECGCNNTransformerClassifier, ClinicalOnlyClassifier
from src.models.explainability import compute_ecg_saliency, compute_clinical_attributions


def test_ecg_encoder_forward():
    encoder = ECGEncoder(in_leads=12, embedding_dim=128)
    dummy_ecg = torch.randn(4, 12, 1000)
    embedding = encoder(dummy_ecg)
    assert embedding.shape == (4, 128)


def test_transformer_encoder_forward():
    transformer = ECGTransformerEncoder(in_leads=12, embedding_dim=128, patch_size=25, num_layers=2, num_heads=4)
    dummy_ecg = torch.randn(2, 12, 1000)
    embedding, tokens, attn = transformer(dummy_ecg, return_temporal_tokens=True)
    assert embedding.shape == (2, 128)
    assert tokens.ndim == 3
    assert attn.ndim == 3


def test_cnn_transformer_hybrid_forward():
    hybrid = ECGCNNTransformerEncoder(in_leads=12, embedding_dim=128, num_heads=4, num_layers=2)
    dummy_ecg = torch.randn(2, 12, 1000)
    embedding = hybrid(dummy_ecg)
    assert embedding.shape == (2, 128)


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


def test_multimodal_transformer_fusion_forward():
    fusion = MultimodalTransformerFusion(ecg_dim=128, clinical_dim=128, out_dim=128, num_heads=4)
    z_ecg = torch.randn(2, 128)
    z_clin = torch.randn(2, 128)
    ecg_tokens = torch.randn(2, 20, 128)
    fused, meta = fusion(z_ecg, z_clin, ecg_tokens=ecg_tokens)
    assert fused.shape == (2, 128)
    assert "ecg_weight" in meta
    assert "clinical_weight" in meta


def test_multimodal_classifier_forward():
    # Test standard Gated CNN-BiLSTM
    model_gated = MultimodalClassifier(in_leads=12, n_clinical_features=10, embedding_dim=128, fusion_type="gated", ecg_backbone="cnn_bilstm")
    ecg = torch.randn(2, 12, 1000)
    clin = torch.randn(2, 10)
    logits, details = model_gated(ecg, clin, return_details=True)
    assert logits.shape == (2,)
    assert "z_ecg" in details
    assert "z_clinical" in details

    # Test Multimodal Transformer
    model_transformer = MultimodalClassifier(
        in_leads=12,
        n_clinical_features=10,
        embedding_dim=128,
        fusion_type="transformer",
        ecg_backbone="transformer"
    )
    logits_t, details_t = model_transformer(ecg, clin, return_details=True)
    assert logits_t.shape == (2,)
    assert "fusion_meta" in details_t

    probs = model_transformer.predict_proba(ecg, clin)
    assert probs.shape == (2,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_baseline_classifiers():
    ecg = torch.randn(2, 12, 1000)
    clin = torch.randn(2, 10)

    # ECG CNN-BiLSTM
    m1 = ECGOnlyClassifier(in_leads=12, embedding_dim=128)
    assert m1.predict_proba(ecg).shape == (2,)

    # ECG Transformer
    m2 = ECGTransformerClassifier(in_leads=12, embedding_dim=128, num_layers=2)
    assert m2.predict_proba(ecg).shape == (2,)

    # ECG CNN-Transformer Hybrid
    m3 = ECGCNNTransformerClassifier(in_leads=12, embedding_dim=128, num_layers=1)
    assert m3.predict_proba(ecg).shape == (2,)

    # Clinical MLP
    m4 = ClinicalOnlyClassifier(n_features=10, embedding_dim=128)
    assert m4.predict_proba(clin).shape == (2,)


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
