import pytest
import torch
from dcis_biomarkers.models.encoders import GenericOmicsEncoder, ClinicalEncoder
from dcis_biomarkers.models.vision_encoder import VisionEncoder
from dcis_biomarkers.models.mil import GatedAttentionMIL
from dcis_biomarkers.models.multimodal_fusion import MultimodalFusionNetwork

def test_omics_encoder():
    encoder = GenericOmicsEncoder(input_dim=100, hidden_dim=32, output_dim=16)
    x = torch.randn(4, 100)
    out = encoder(x)
    assert out.shape == (4, 16)

def test_clinical_encoder():
    encoder = ClinicalEncoder(input_dim=10, output_dim=8)
    x = torch.randn(4, 10)
    out = encoder(x)
    assert out.shape == (4, 8)

def test_gated_attention_mil():
    mil = GatedAttentionMIL(input_dim=64, hidden_dim=32)
    x = torch.randn(2, 10, 64) # Batch=2, Bag=10, Dim=64
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 5:] = False # Primer elemento tiene 5 instancias válidas
    
    fused, attn = mil(x, mask)
    
    assert fused.shape == (2, 64)
    assert attn.shape == (2, 10)
    
    # Comprobar que la atención es 0 para instancias enmascaradas
    assert torch.allclose(attn[0, 5:], torch.zeros(5))
    assert torch.allclose(attn[0, :5].sum(), torch.tensor(1.0))

def test_multimodal_fusion_network():
    net = MultimodalFusionNetwork(
        omics_dims={"transcriptomics": 100, "clinical": 10},
        vision_dim=64,
        num_classes=2,
        fused_dim=32
    )
    
    vision_feats = torch.randn(2, 15, 64)
    vision_mask = torch.ones(2, 15, dtype=torch.bool)
    omics_dict = {
        "transcriptomics": torch.randn(2, 100),
        "clinical": torch.randn(2, 10)
    }
    
    out = net(vision_feats, vision_mask, omics_dict)
    
    assert "logits" in out
    assert out["logits"].shape == (2, 2)
    assert "roi_attention_weights" in out
    assert out["roi_attention_weights"].shape == (2, 15)
