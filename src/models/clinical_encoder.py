"""
Clinical Feature Extractor using MLP with LayerNorm, GELU, and Dropout.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 128,
        hidden_dims: tuple = (64, 96),
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim

        layers.extend([
            nn.Linear(in_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, clinical_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, n_features]
        Output: embedding [batch, embedding_dim]
        """
        return self.net(clinical_tensor)
