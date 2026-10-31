"""
Cross-Modal Fusion Modules:
- ConcatFusion (Baseline)
- GatedFusion (Dynamic Modality Gating)
- CrossAttentionFusion (Bidirectional / Asymmetric Cross-Modal Attention)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Simple concatenation of modality embeddings."""
    def __init__(self, ecg_dim: int = 128, clinical_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ecg_dim + clinical_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        self.out_dim = out_dim

    def forward(self, z_ecg: torch.Tensor, z_clinical: torch.Tensor):
        cat = torch.cat([z_ecg, z_clinical], dim=-1)
        return self.proj(cat), None


class GatedFusion(nn.Module):
    """
    Gated Multimodal Fusion.
    Learns dynamic scalar or vector gating factors representing modality confidence:
    g = sigmoid(W_g * [z_ecg; z_clinical] + b_g)
    fused = [g * z_ecg; (1 - g) * z_clinical]
    """
    def __init__(self, ecg_dim: int = 128, clinical_dim: int = 128, out_dim: int = 128):
        super().__init__()
        # Gate generator
        self.gate_net = nn.Sequential(
            nn.Linear(ecg_dim + clinical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, ecg_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(ecg_dim + clinical_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        self.out_dim = out_dim

    def forward(self, z_ecg: torch.Tensor, z_clinical: torch.Tensor):
        cat = torch.cat([z_ecg, z_clinical], dim=-1)
        gate = self.gate_net(cat)  # [batch, ecg_dim] in range [0, 1]

        # Modulated representations
        gated_ecg = gate * z_ecg
        gated_clinical = (1.0 - gate) * z_clinical

        fused = self.proj(torch.cat([gated_ecg, gated_clinical], dim=-1))
        # Compute scalar summary of modality confidence for explainability/audit
        ecg_contrib = float(gate.mean().item())
        clinical_contrib = 1.0 - ecg_contrib
        modality_weights = {"ecg_weight": round(ecg_contrib, 3), "clinical_weight": round(clinical_contrib, 3)}

        return fused, modality_weights


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion where clinical features attend to temporal ECG tokens.
    """
    def __init__(self, ecg_dim: int = 128, clinical_dim: int = 128, out_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=ecg_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(ecg_dim)
        self.norm2 = nn.LayerNorm(clinical_dim)
        self.proj = nn.Sequential(
            nn.Linear(ecg_dim + clinical_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        self.out_dim = out_dim

    def forward(self, z_ecg: torch.Tensor, z_clinical: torch.Tensor, ecg_tokens: torch.Tensor = None):
        """
        If ecg_tokens [batch, time_steps, dim] are provided, query with clinical tokens.
        Otherwise cross-attend between global embeddings.
        """
        if ecg_tokens is not None:
            q = z_clinical.unsqueeze(1)  # [batch, 1, dim]
            kv = ecg_tokens             # [batch, time, dim]
            attn_out, attn_weights = self.mha(q, kv, kv)
            attended_ecg = attn_out.squeeze(1)
        else:
            q = z_clinical.unsqueeze(1)
            kv = z_ecg.unsqueeze(1)
            attn_out, attn_weights = self.mha(q, kv, kv)
            attended_ecg = attn_out.squeeze(1)

        fused = self.proj(torch.cat([attended_ecg, z_clinical], dim=-1))
        return fused, {"cross_attention": True}


def get_fusion_module(fusion_type: str, ecg_dim: int = 128, clinical_dim: int = 128, out_dim: int = 128) -> nn.Module:
    fusion_type = fusion_type.lower()
    if fusion_type == "gated":
        return GatedFusion(ecg_dim, clinical_dim, out_dim)
    elif fusion_type in ("cross_attention", "attention"):
        return CrossAttentionFusion(ecg_dim, clinical_dim, out_dim)
    elif fusion_type in ("concat", "concatenation"):
        return ConcatFusion(ecg_dim, clinical_dim, out_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Choose 'gated', 'concat', or 'cross_attention'.")
