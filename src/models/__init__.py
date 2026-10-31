"""Models package including neural encoders, Transformer, fusion, baselines, explainability, and artifacts."""
from src.models.ecg_encoder import ECGEncoder, MultiScaleConv1D, ConvBlock1D
from src.models.transformer_encoder import ECGTransformerEncoder, ECGCNNTransformerEncoder, PositionalEncoding
from src.models.clinical_encoder import ClinicalEncoder
from src.models.fusion import (
    ConcatFusion,
    GatedFusion,
    CrossAttentionFusion,
    MultimodalTransformerFusion,
    get_fusion_module
)
from src.models.multimodal_classifier import MultimodalClassifier
from src.models.baselines import (
    ECGOnlyClassifier,
    ECGTransformerClassifier,
    ECGCNNTransformerClassifier,
    ClinicalOnlyClassifier
)
from src.models.explainability import compute_ecg_saliency, compute_clinical_attributions
from src.models.trainer import MultimodalTrainer, compute_binary_metrics, find_optimal_threshold
from src.models.artifact import ModelArtifactManager

__all__ = [
    "ECGEncoder",
    "MultiScaleConv1D",
    "ConvBlock1D",
    "ECGTransformerEncoder",
    "ECGCNNTransformerEncoder",
    "PositionalEncoding",
    "ClinicalEncoder",
    "ConcatFusion",
    "GatedFusion",
    "CrossAttentionFusion",
    "MultimodalTransformerFusion",
    "get_fusion_module",
    "MultimodalClassifier",
    "ECGOnlyClassifier",
    "ECGTransformerClassifier",
    "ECGCNNTransformerClassifier",
    "ClinicalOnlyClassifier",
    "compute_ecg_saliency",
    "compute_clinical_attributions",
    "MultimodalTrainer",
    "compute_binary_metrics",
    "find_optimal_threshold",
    "ModelArtifactManager"
]
