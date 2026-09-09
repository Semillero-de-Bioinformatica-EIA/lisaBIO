import torch
import torch.nn as nn
from typing import Dict, Tuple

from .encoders import GenericOmicsEncoder, ClinicalEncoder
from .mil import GatedAttentionMIL

class MultimodalFusionNetwork(nn.Module):
    """
    Red de Fusión Multimodal Refactorizada.
    Acepta cualquier número de modalidades ómicas, procesa WSI con MIL,
    e integra con un gating network.
    """
    def __init__(self, 
                 omics_dims: Dict[str, int], 
                 vision_dim: int,
                 num_classes: int = 2,
                 fused_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.omics_dims = omics_dims
        
        # Encoders ómicos
        self.omics_encoders = nn.ModuleDict()
        for mod, dim in omics_dims.items():
            if mod == "clinical":
                self.omics_encoders[mod] = ClinicalEncoder(dim)
            else:
                self.omics_encoders[mod] = GenericOmicsEncoder(dim)
                
        # Vision MIL
        self.vision_mil = GatedAttentionMIL(input_dim=vision_dim)
        
        # Gating network para fusión (Vision + Suma de Omics)
        # En una arquitectura más avanzada esto podría ser Cross-Attention
        self.gate = nn.Sequential(
            nn.Linear(vision_dim + sum([e.net[-2].out_features if hasattr(e.net[-2], 'out_features') else 256 for e in self.omics_encoders.values()]), 2),
            nn.Softmax(dim=1)
        )
        
        fused_input_dim = vision_dim + sum([e.net[-2].out_features if hasattr(e.net[-2], 'out_features') else 256 for e in self.omics_encoders.values()])
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_input_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes)
        )
        
        # Proportional Hazards head for Cox loss (optional output)
        self.hazard_head = nn.Linear(fused_dim, 1)

    def forward(self, vision_features: torch.Tensor, vision_mask: torch.Tensor, omics_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        vision_features: (B, N, D_v)
        vision_mask: (B, N) boolean
        omics_dict: {"transcriptomics": (B, D_t), "clinical": (B, D_c)}
        """
        # 1. Vision MIL
        v_fused, v_attn = self.vision_mil(vision_features, vision_mask)
        
        # 2. Omics Encoding
        o_features = []
        for mod, tensor in omics_dict.items():
            encoded = self.omics_encoders[mod](tensor)
            o_features.append(encoded)
            
        o_fused = torch.cat(o_features, dim=1) if o_features else torch.empty(v_fused.size(0), 0, device=v_fused.device)
        
        # 3. Concatenation & Gating
        concat_features = torch.cat([v_fused, o_fused], dim=1)
        gate_weights = self.gate(concat_features) # (B, 2)
        
        # Aplicar pesos de la compuerta (asumiendo gate_weights[0] para vision, [1] para omics combinadas)
        # Esto es un pseudo-gating simple para demostración
        v_gated = v_fused * gate_weights[:, 0].unsqueeze(1)
        o_gated = o_fused * gate_weights[:, 1].unsqueeze(1)
        
        final_fused = torch.cat([v_gated, o_gated], dim=1)
        
        # 4. Clasificación
        logits = self.classifier(final_fused)
        
        # 5. Supervivencia
        # Extraer activaciones de la penúltima capa
        hidden = self.classifier[:-1](final_fused)
        hazard = self.hazard_head(hidden)
        
        return {
            "logits": logits,
            "hazard_risk": hazard,
            "roi_attention_weights": v_attn,
            "gate_weights": gate_weights,
            "fused_representation": hidden
        }
