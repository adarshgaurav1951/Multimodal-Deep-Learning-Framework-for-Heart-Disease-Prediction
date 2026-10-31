"""
Unimodal Baselines for Benchmarking:
- ECGOnlyClassifier (1D CNN + BiLSTM + Attention)
- ECGTransformerClassifier (Patch Tokenization + Self-Attention Transformer)
- ECGCNNTransformerClassifier (Multi-Scale CNN + Transformer Hybrid)
- ClinicalOnlyClassifier (Tabular MLP)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from src.models.ecg_encoder import ECGEncoder
from src.models.transformer_encoder import ECGTransformerEncoder, ECGCNNTransformerEncoder
from src.models.clinical_encoder import ClinicalEncoder


class ECGOnlyClassifier(nn.Module):
    """ECG-only baseline using Multi-scale CNN + BiLSTM + Temporal Attention."""
    def __init__(self, in_leads: int = 12, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = ECGEncoder(in_leads=in_leads, embedding_dim=embedding_dim, dropout=dropout * 0.7)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        z = self.encoder(ecg)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, ecg: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(ecg))


class ECGTransformerClassifier(nn.Module):
    """ECG-only baseline using Pure Transformer Encoder with Self-Attention."""
    def __init__(
        self,
        in_leads: int = 12,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = ECGTransformerEncoder(
            in_leads=in_leads,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        z = self.encoder(ecg)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, ecg: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(ecg))


class ECGCNNTransformerClassifier(nn.Module):
    """ECG-only baseline using Hybrid Multi-Scale CNN + Transformer."""
    def __init__(
        self,
        in_leads: int = 12,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = ECGCNNTransformerEncoder(
            in_leads=in_leads,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        z = self.encoder(ecg)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, ecg: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(ecg))


class ClinicalOnlyClassifier(nn.Module):
    """Clinical-only baseline using Deep Tabular MLP."""
    def __init__(self, n_features: int, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = ClinicalEncoder(n_features=n_features, embedding_dim=embedding_dim, dropout=dropout * 0.7)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        z = self.encoder(clinical)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, clinical: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(clinical))
