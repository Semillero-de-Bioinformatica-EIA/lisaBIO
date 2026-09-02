import pytest
import torch
from dcis_biomarkers.models import MultimodalFusionNetwork, OmicsEncoder, WSIBagEncoder

def test_omics_encoder():
    encoder = OmicsEncoder(input_dim=100, output_dim=32)
    x = torch.randn(4, 100)
    out = encoder(x)
    assert out.shape == (4, 32)

def test_wsi_bag_encoder():
    encoder = WSIBagEncoder(input_dim=64, output_dim=32)
    patches = torch.randn(10, 64)
    out = encoder(patches)
    assert out.shape == (1, 32)

def test_multimodal_fusion_network():
    net = MultimodalFusionNetwork(omics_dim=100, vision_dim=64, latent_dim=32, num_classes=2)
    omics_data = torch.randn(1, 100)
    wsi_patches = torch.randn(15, 64)
    logits, fused_rep, gate = net(omics_data, wsi_patches)
    assert logits.shape == (1, 2)
    assert fused_rep.shape == (1, 32)
