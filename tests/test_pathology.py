import pytest
import numpy as np
from dcis_biomarkers.pathology import WSITiler, segment_tissue, build_spatial_microenvironment_graph

def test_wsi_tiler():
    tiler = WSITiler(patch_size=256, stride=256)
    coords = tiler.extract_patches_coordinates((1024, 1024))
    assert len(coords) == 16

def test_tissue_segmentation():
    dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8) # Fondo blanco
    dummy_img[20:80, 20:80] = [150, 50, 150] # Tejido H&E rosado/púrpura
    mask = segment_tissue(dummy_img)
    assert mask[50, 50] == 1
    assert mask[0, 0] == 0

def test_spatial_graph():
    coords = np.array([[0, 0], [10, 10], [20, 20], [100, 100]])
    feats = np.random.rand(4, 32)
    graph = build_spatial_microenvironment_graph(coords, feats, k_neighbors=2)
    assert "x" in graph
    assert "edge_index" in graph
