"""
ECG Transformer Encoders:
1. PositionalEncoding: Sinusoidal and learnable positional encodings for ECG sequences.
2. ECGTransformerEncoder: Pure Transformer architecture with patch tokenization and multi-head self-attention.
3. ECGCNNTransformerEncoder: Hybrid architecture combining Multi-Scale 1D CNN with Transformer self-attention.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ecg_encoder import MultiScaleConv1D, ConvBlock1D


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal tokens."""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ECGTransformerEncoder(nn.Module):
    """
    Pure Transformer Architecture for 12-Lead ECG Signals.
    - Tokenizes multi-lead time series using strided patch convolution (capturing 50ms cardiac segments).
    - Models beat-to-beat and long-range dependencies via multi-head self-attention layers.
    - Applies temporal attention pooling to extract fixed-size cardiovascular representation.
    """
    def __init__(
        self,
        in_leads: int = 12,
        embedding_dim: int = 128,
        patch_size: int = 25,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.15
    ):
        super().__init__()
        self.in_leads = in_leads
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size

        # Patch Tokenizer: 1D Conv across all leads
        # 2500 samples / 25 stride = 100 temporal tokens (each covering 50ms at 500Hz)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_leads, embedding_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, max_len=500, dropout=dropout)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Temporal attention pooling over tokens
        self.attention_pool = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Final projection head
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_temporal_tokens: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: [batch, in_leads, samples]
        Output: z_ecg [batch, embedding_dim], optional temporal tokens & attention weights
        """
        # Patch embedding: [batch, embedding_dim, seq_len]
        tokens = self.patch_embed(x)
        tokens = tokens.transpose(1, 2)  # [batch, seq_len, embedding_dim]

        # Add positional encoding
        tokens = self.pos_encoder(tokens)

        # Transformer self-attention pass: [batch, seq_len, embedding_dim]
        encoded = self.transformer_encoder(tokens)

        # Temporal attention pooling
        attn_scores = self.attention_pool(encoded)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
        pooled = torch.sum(encoded * attn_weights, dim=1)  # [batch, embedding_dim]

        z_ecg = self.proj(pooled)

        if return_temporal_tokens:
            return z_ecg, encoded, attn_weights
        return z_ecg


class ECGCNNTransformerEncoder(nn.Module):
    """
    Hybrid Multi-Scale CNN + Transformer Encoder.
    - Multi-scale CNN captures high-frequency localized morphological transients (QRS complexes).
    - Multi-head Self-Attention Transformer captures rhythm, sequence progression, and repolarization dynamics.
    - Combines inductive bias of convolutions with global receptive field of self-attention.
    """
    def __init__(
        self,
        in_leads: int = 12,
        embedding_dim: int = 128,
        cnn_channels: tuple = (32, 64, 128),
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.15
    ):
        super().__init__()
        self.in_leads = in_leads
        self.embedding_dim = embedding_dim

        # CNN Feature Extraction Frontend
        self.stage1 = MultiScaleConv1D(in_leads, cnn_channels[0])
        self.stage2 = ConvBlock1D(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=2, pool=True)
        self.stage3 = ConvBlock1D(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=1, pool=True)

        # Linear projection to Transformer dimension
        self.proj_in = nn.Linear(cnn_channels[2], embedding_dim)
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, max_len=500, dropout=dropout)

        # Transformer Temporal Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

        self.proj_out = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_temporal_tokens: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # CNN forward: [batch, cnn_channels[2], time_steps]
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)

        # Reshape for Transformer: [batch, time_steps, channels]
        h = h.transpose(1, 2)
        h = self.proj_in(h)
        h = self.pos_encoder(h)

        # Self-attention over temporal tokens
        trans_out = self.transformer(h)  # [batch, time_steps, embedding_dim]

        # Temporal attention pooling
        attn_weights = F.softmax(self.attention(trans_out), dim=1)  # [batch, time_steps, 1]
        pooled = torch.sum(trans_out * attn_weights, dim=1)  # [batch, embedding_dim]

        z_ecg = self.proj_out(pooled)

        if return_temporal_tokens:
            return z_ecg, trans_out, attn_weights
        return z_ecg
