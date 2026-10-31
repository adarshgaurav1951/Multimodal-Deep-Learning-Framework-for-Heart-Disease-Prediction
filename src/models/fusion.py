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
        # Compute relative norm energy for telemetry
        ecg_n = float(torch.norm(attended_ecg, dim=-1).mean().item())
        clin_n = float(torch.norm(z_clinical, dim=-1).mean().item())
        total_n = ecg_n + clin_n if (ecg_n + clin_n) > 0 else 1.0
        ecg_w = round(ecg_n / total_n, 3)

        return fused, {
            "cross_attention": True,
            "ecg_weight": ecg_w,
            "clinical_weight": round(1.0 - ecg_w, 3)
        }


class MultimodalTransformerFusion(nn.Module):
    """
    Multimodal Transformer Joint Attention Fusion.
    Concatenates clinical token and ECG temporal tokens into a joint cross-modal sequence,
    allowing full bidirectional self-attention between modalities.
    """
    def __init__(self, ecg_dim: int = 128, clinical_dim: int = 128, out_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.clin_proj = nn.Linear(clinical_dim, ecg_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ecg_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.joint_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.proj = nn.Sequential(
            nn.Linear(ecg_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        self.out_dim = out_dim

    def forward(self, z_ecg: torch.Tensor, z_clinical: torch.Tensor, ecg_tokens: torch.Tensor = None):
        clin_token = self.clin_proj(z_clinical).unsqueeze(1)  # [batch, 1, ecg_dim]
        if ecg_tokens is not None:
            joint_seq = torch.cat([clin_token, ecg_tokens], dim=1)  # [batch, 1 + seq_len, dim]
        else:
            joint_seq = torch.cat([clin_token, z_ecg.unsqueeze(1)], dim=1)

        joint_out = self.joint_transformer(joint_seq)
        attended_clin = joint_out[:, 0, :]
        attended_ecg = torch.mean(joint_out[:, 1:, :], dim=1)

        fused = self.proj(torch.cat([attended_ecg, attended_clin], dim=-1))

        ecg_n = float(torch.norm(attended_ecg, dim=-1).mean().item())
        clin_n = float(torch.norm(attended_clin, dim=-1).mean().item())
        total_n = ecg_n + clin_n if (ecg_n + clin_n) > 0 else 1.0
        ecg_w = round(ecg_n / total_n, 3)

        return fused, {
            "transformer_fusion": True,
            "ecg_weight": ecg_w,
            "clinical_weight": round(1.0 - ecg_w, 3)
        }


def get_fusion_module(fusion_type: str, ecg_dim: int = 128, clinical_dim: int = 128, out_dim: int = 128) -> nn.Module:
    fusion_type = fusion_type.lower()
    if fusion_type == "gated":
        return GatedFusion(ecg_dim, clinical_dim, out_dim)
    elif fusion_type in ("cross_attention", "attention"):
        return CrossAttentionFusion(ecg_dim, clinical_dim, out_dim)
    elif fusion_type in ("transformer", "transformer_fusion", "multimodal_transformer"):
        return MultimodalTransformerFusion(ecg_dim, clinical_dim, out_dim)
    elif fusion_type in ("concat", "concatenation"):
        return ConcatFusion(ecg_dim, clinical_dim, out_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Choose 'gated', 'concat', 'cross_attention', or 'transformer'.")
