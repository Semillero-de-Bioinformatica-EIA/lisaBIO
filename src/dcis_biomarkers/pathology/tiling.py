from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image
import numpy as np

from .slide_reader import SlideReader
from .tissue_segmentation import segment_tissue

class WSITiler:
    """
    Genera y extrae parches (tiles) de una Whole Slide Image (WSI) en
    un nivel específico, basándose en la resolución física o dimensiones del parche.
    """
    def __init__(self, patch_size: int = 256, stride: int = 256, target_mpp: float = 0.5):
        self.patch_size = patch_size
        self.stride = stride
        self.target_mpp = target_mpp

    def _determine_best_level(self, reader: SlideReader) -> int:
        # Simplificación: usar nivel 0 o aproximar al target_mpp si los mpps están disponibles
        mpp_x, _ = reader.mpp
        if mpp_x is None:
            return 0
        
        # TODO: Lógica real para encontrar el nivel más cercano a target_mpp
        return 0

    def extract_patches_coordinates(self, dimensions: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Calcula las coordenadas superiores izquierdas (x, y) de todos los parches posibles.
        """
        w, h = dimensions
        coords = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                coords.append((x, y))
        return coords

    def tile_slide(self, slide_path: str | Path, min_tissue_ratio: float = 0.5) -> List[Dict]:
        """
        Extrae parches validando la cantidad de tejido. Retorna metadata de los parches válidos.
        """
        reader = SlideReader(slide_path)
        level = self._determine_best_level(reader)
        
        # Asumimos extraer del nivel 0 para coordenadas
        w, h = reader.dimensions
        coords = self.extract_patches_coordinates((w, h))
        
        valid_patches_meta = []
        
        for (x, y) in coords:
            img = reader.read_region((x, y), level, (self.patch_size, self.patch_size))
            img_arr = np.array(img)
            
            mask, qc = segment_tissue(img_arr)
            if qc["tissue_percentage"] >= min_tissue_ratio * 100:
                valid_patches_meta.append({
                    "x": x,
                    "y": y,
                    "level": level,
                    "tissue_percentage": qc["tissue_percentage"]
                })
                
        reader.close()
        return valid_patches_meta
