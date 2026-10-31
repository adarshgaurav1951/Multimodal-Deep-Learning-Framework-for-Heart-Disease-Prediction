"""Models package including neural encoders, fusion, baselines, explainability, and artifacts."""
from src.models.ecg_encoder import ECGEncoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.fusion import ConcatFusion, GatedFusion, CrossAttentionFusion, get_fusion_module
from src.models.multimodal_classifier import MultimodalClassifier
from src.models.baselines import ECGOnlyClassifier, ClinicalOnlyClassifier
from src.models.explainability import compute_ecg_saliency, compute_clinical_attributions
from src.models.trainer import MultimodalTrainer, compute_binary_metrics, find_optimal_threshold
from src.models.artifact import ModelArtifactManager

__all__ = [
    "ECGEncoder",
    "ClinicalEncoder",
    "ConcatFusion",
    "GatedFusion",
    "CrossAttentionFusion",
    "get_fusion_module",
    "MultimodalClassifier",
    "ECGOnlyClassifier",
    "ClinicalOnlyClassifier",
    "compute_ecg_saliency",
    "compute_clinical_attributions",
    "MultimodalTrainer",
    "compute_binary_metrics",
    "find_optimal_threshold",
    "ModelArtifactManager"
]
