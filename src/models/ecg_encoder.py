"""
ECG Feature Extractor using Multi-Scale 1D CNN and Bidirectional LSTM.
Captures both high-frequency localized morphological transients (QRS complexes)
and low-frequency global rhythm and repolarization patterns (ST segments, T-waves).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, pool: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class MultiScaleConv1D(nn.Module):
    """
    Parallel multi-scale 1D convolution block with kernels 3, 7, and 15.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c_branch = out_channels // 3
        c_last = out_channels - (c_branch * 2)

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, c_branch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_branch),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, c_branch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c_branch),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c_last, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(c_last),
            nn.GELU()
        )
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        cat = torch.cat([b1, b2, b3], dim=1)
        return self.pool(cat)


class ECGEncoder(nn.Module):
    def __init__(
        self,
        in_leads: int = 12,
        embedding_dim: int = 128,
        cnn_channels: tuple = (32, 64, 128),
        lstm_hidden: int = 64,
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_leads = in_leads
        self.embedding_dim = embedding_dim

        # Stage 1: Initial multi-scale convolution
        self.stage1 = MultiScaleConv1D(in_leads, cnn_channels[0])

        # Stage 2: Intermediate conv block
        self.stage2 = ConvBlock1D(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=2, pool=True)

        # Stage 3: Deep conv block
        self.stage3 = ConvBlock1D(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=1, pool=True)

        # BiLSTM temporal encoder
        lstm_input_dim = cnn_channels[2]
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden

        # Temporal attention pooling
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Final projection to embedding space
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, return_temporal_tokens: bool = False):
        """
        Input: [batch, in_leads, samples]
        Output: embedding [batch, embedding_dim], optional temporal tokens [batch, time_steps, lstm_out_dim]
        """
        # CNN forward
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)  # [batch, cnn_channels[2], time_steps]

        # Reshape for LSTM: [batch, time_steps, cnn_channels[2]]
        h = h.transpose(1, 2)
        lstm_out, _ = self.lstm(h)  # [batch, time_steps, lstm_out_dim]

        # Attention pooling over time
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch, time_steps, 1]
        pooled = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, lstm_out_dim]

        # Projection
        z_ecg = self.proj(pooled)  # [batch, embedding_dim]

        if return_temporal_tokens:
            return z_ecg, lstm_out, attn_weights
        return z_ecg
