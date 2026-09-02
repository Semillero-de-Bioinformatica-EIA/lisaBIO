import numpy as np
from typing import List, Tuple, Dict, Any

class WSITiler:
    """Dividir láminas completas de histopatología (WSI) en parches (tiles) informativos."""

    def __init__(self, patch_size: int = 256, stride: int = 256, target_mag: str = "20x"):
        self.patch_size = patch_size
        self.stride = stride
        self.target_mag = target_mag

    def extract_patches_coordinates(self, image_dimensions: Tuple[int, int], tissue_mask: np.ndarray = None) -> List[Tuple[int, int]]:
        """Calcula coordenadas (x, y) de los parches dentro del área con tejido histológico."""
        width, height = image_dimensions
        coords = []

        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                if tissue_mask is not None:
                    # Verificar si la zona contiene suficiente proporción de tejido
                    mask_patch = tissue_mask[y:y+self.patch_size, x:x+self.patch_size]
                    if np.mean(mask_patch > 0) < 0.2:
                        continue
                coords.append((x, y))

        return coords
