import pytest
import numpy as np
from dcis_biomarkers.pathology.tissue_segmentation import segment_tissue
from dcis_biomarkers.pathology.tiling import WSITiler
from dcis_biomarkers.pathology.spatial_graph import build_spatial_microenvironment_graph

def test_tissue_segmentation():
    # Imagen dummy RGB (H, W, 3)
    dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8) # Blanco (fondo)
    # Simular tejido (rosado/púrpura)
    dummy_img[20:80, 20:80] = [200, 100, 150]
    
    mask, qc = segment_tissue(dummy_img)
    
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 1
    assert mask[0, 0] == 0
    assert qc["tissue_percentage"] > 20.0

def test_wsi_tiler_coordinates():
    tiler = WSITiler(patch_size=256, stride=256)
    coords = tiler.extract_patches_coordinates((1024, 1024))
    # 1024 / 256 = 4 -> 4x4 = 16 patches
    assert len(coords) == 16

def test_spatial_graph():
    # Coordenadas en micrones
    coords = np.array([
        [0, 0],
        [50, 0],   # a 50 um
        [150, 0],  # a 150 um (debe estar desconectado si threshold=100)
    ])
    features = np.random.randn(3, 10)
    
    graph = build_spatial_microenvironment_graph(coords, features, k_neighbors=2, distance_threshold_um=100.0)
    
    edges = graph["edge_index"].T.tolist()
    assert [0, 1] in edges or [1, 0] in edges
    assert [0, 2] not in edges
    assert [2, 0] not in edges
