"""
Cross-Attention Multimodal Fusion Network - v5.
Improvements:
  - Temperature-smoothed Adaptive Gating (gate_temperature = 1.5) to prevent binary 1.0/0.0 collapse
  - Differential LR support (Visual Backbone lr_vision = 1e-5, Rest lr = 1e-4)
  - Focal Loss compatibility
  - Softmax temperature scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout: float = 0.1, gate_temperature: float = 1.5):
        super().__init__()
        self.gate_temperature = gate_temperature
        self.cross_attn_v2o = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_o2v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.gate_fc = nn.Linear(embed_dim * 2, 2)

    def forward(self, v_feat: torch.Tensor, o_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_attended, v_weights = self.cross_attn_v2o(query=v_feat, key=o_feat, value=o_feat)
        v_out = self.norm1(v_feat + v_attended)

        o_attended, o_weights = self.cross_attn_o2v(query=o_feat, key=v_feat, value=v_feat)
        o_out = self.norm2(o_feat + o_attended)

        v_sq = v_out.squeeze(1)
        o_sq = o_out.squeeze(1)

        combined = torch.cat([v_sq, o_sq], dim=-1)
        
        # Temperature-smoothed gating to prevent binary 1.0 / 0.0 collapse
        gate_logits = self.gate_fc(combined) / self.gate_temperature
        gate_weights = F.softmax(gate_logits, dim=-1)

        fused = gate_weights[:, 0:1] * v_sq + gate_weights[:, 1:2] * o_sq
        fused = self.mlp(torch.cat([fused, fused], dim=-1))

        return fused, v_weights, gate_weights


class MultimodalFusionNetwork(nn.Module):
    def __init__(
        self,
        omics_input_dim: int = 1000,
        vision_embed_dim: int = 512,
        omics_embed_dim: int = 512,
        fused_dim: int = 512,
        num_classes: int = 3,
        use_monai: bool = True,
        freeze_backbone: bool = False,
        temperature: float = 0.5,
        gate_temperature: float = 1.5
    ):
        super().__init__()
        from .cnn_rnn_monai import MONAIPathologyCNNRNNModel
        from .omics_encoder import OmicsEncoder

        self.pathology_model = MONAIPathologyCNNRNNModel(
            num_classes=num_classes,
            cnn_feature_dim=vision_embed_dim,
            rnn_hidden_dim=vision_embed_dim // 2,
            use_monai=use_monai,
            max_seq_len=20,
            freeze_backbone=freeze_backbone,
            temperature=temperature
        )

        self.omics_encoder = OmicsEncoder(
            input_dim=omics_input_dim,
            hidden_dim=omics_embed_dim // 2,
            output_dim=omics_embed_dim
        )

        self.vision_proj = nn.Sequential(nn.Linear(vision_embed_dim, fused_dim), nn.LayerNorm(fused_dim))
        self.omics_proj  = nn.Sequential(nn.Linear(omics_embed_dim,  fused_dim), nn.LayerNorm(fused_dim))

        self.fusion_block = CrossAttentionFusionBlock(
            embed_dim=fused_dim, num_heads=4, dropout=0.1, gate_temperature=gate_temperature
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.survival_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(
        self,
        patch_sequences: torch.Tensor,
        omic_vectors: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pathology_out  = self.pathology_model(patch_sequences)
        v_raw          = pathology_out["sequence_embeddings"]
        roi_attention  = pathology_out["attention_weights"]

        o_raw = self.omics_encoder(omic_vectors)

        v_proj = self.vision_proj(v_raw).unsqueeze(1)
        o_proj = self.omics_proj(o_raw).unsqueeze(1)

        fused_embedding, cross_attn_w, gate_w = self.fusion_block(v_proj, o_proj)

        logits       = self.classifier(fused_embedding)
        hazard_risk  = self.survival_head(fused_embedding)

        return {
            "logits":                 logits,
            "hazard_risk":            hazard_risk,
            "fused_embedding":        fused_embedding,
            "roi_attention_weights":  roi_attention,
            "cross_attention_weights": cross_attn_w,
            "gate_weights":           gate_w
        }
