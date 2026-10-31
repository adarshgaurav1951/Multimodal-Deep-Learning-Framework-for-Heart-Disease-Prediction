"""
Unified Multimodal Classifier.
Combines ECGEncoder (CNN-BiLSTM / Transformer / CNN-Transformer) + ClinicalEncoder (MLP)
+ Multimodal Fusion (Gated / Cross-Attention / Transformer / Concat) + Classification Head.
"""
from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from src.models.ecg_encoder import ECGEncoder
from src.models.transformer_encoder import ECGTransformerEncoder, ECGCNNTransformerEncoder
from src.models.clinical_encoder import ClinicalEncoder
from src.models.fusion import get_fusion_module


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        in_leads: int = 12,
        n_clinical_features: int = 12,
        embedding_dim: int = 128,
        fusion_type: str = "gated",
        ecg_backbone: str = "cnn_bilstm",
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_leads = in_leads
        self.n_clinical_features = n_clinical_features
        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type
        self.ecg_backbone = ecg_backbone.lower()

        # ECG Encoder selection
        if self.ecg_backbone in ("transformer", "vit", "transformer_encoder"):
            self.ecg_encoder = ECGTransformerEncoder(
                in_leads=in_leads,
                embedding_dim=embedding_dim,
                dropout=dropout * 0.7
            )
        elif self.ecg_backbone in ("cnn_transformer", "hybrid"):
            self.ecg_encoder = ECGCNNTransformerEncoder(
                in_leads=in_leads,
                embedding_dim=embedding_dim,
                dropout=dropout * 0.7
            )
        else:
            # Default: Multi-Scale 1D CNN + BiLSTM + Attention
            self.ecg_encoder = ECGEncoder(
                in_leads=in_leads,
                embedding_dim=embedding_dim,
                dropout=dropout * 0.7
            )

        # Clinical tabular encoder
        self.clinical_encoder = ClinicalEncoder(
            n_features=n_clinical_features,
            embedding_dim=embedding_dim,
            dropout=dropout * 0.7
        )

        # Multimodal fusion module
        self.fusion = get_fusion_module(
            fusion_type,
            ecg_dim=embedding_dim,
            clinical_dim=embedding_dim,
            out_dim=embedding_dim
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        ecg: torch.Tensor,
        clinical: torch.Tensor,
        return_details: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        """
        ecg: [batch, in_leads, samples]
        clinical: [batch, n_clinical_features]
        
        returns logits [batch] or (logits [batch], details dict)
        """
        # ECG branch
        if self.fusion_type in ("cross_attention", "attention", "transformer", "transformer_fusion", "multimodal_transformer"):
            z_ecg, ecg_tokens, attn_weights = self.ecg_encoder(ecg, return_temporal_tokens=True)
            z_clinical = self.clinical_encoder(clinical)
            fused, fusion_meta = self.fusion(z_ecg, z_clinical, ecg_tokens=ecg_tokens)
        else:
            z_ecg = self.ecg_encoder(ecg)
            z_clinical = self.clinical_encoder(clinical)
            fused, fusion_meta = self.fusion(z_ecg, z_clinical)

        # Projection to logit
        logits = self.classifier(fused).squeeze(-1)  # [batch]

        if return_details:
            details = {
                "z_ecg": z_ecg,
                "z_clinical": z_clinical,
                "fused": fused,
                "fusion_meta": fusion_meta or {}
            }
            return logits, details

        return logits

    @torch.no_grad()
    def predict_proba(self, ecg: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """Convenience method returning sigmoid probabilities."""
        self.eval()
        logits = self.forward(ecg, clinical)
        return torch.sigmoid(logits)
