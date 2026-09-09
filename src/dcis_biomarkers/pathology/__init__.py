"""
dcis_biomarkers.pathology
-------------------------
Módulo para el procesamiento robusto de Whole Slide Images (WSI),
segmentación de tejido, y extracción de parches.
"""

from .slide_reader import SlideReader
from .tissue_segmentation import segment_tissue
from .tiling import WSITiler
from .stain_normalization import StainNormalizer
from .augmentations import get_train_augmentations, get_eval_augmentations

__all__ = [
    "SlideReader",
    "segment_tissue",
    "WSITiler",
    "StainNormalizer",
    "get_train_augmentations",
    "get_eval_augmentations",
]
