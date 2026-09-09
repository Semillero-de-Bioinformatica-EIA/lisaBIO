import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None

try:
    import tifffile
except ImportError:
    tifffile = None

class SlideReader:
    """
    Lector de Whole Slide Images (WSI) agnóstico al formato (SVS, NDPI, TIFF, OME-TIFF).
    Usa OpenSlide como backend principal y tifffile como fallback para TIFFs estándar.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"WSI no encontrada en {self.path}")

        self.backend = None
        self._slide_os = None
        self._slide_tf = None
        self._initialize_reader()

    def _initialize_reader(self):
        # Intentar con OpenSlide primero
        if openslide is not None and openslide.OpenSlide.detect_format(str(self.path)) is not None:
            try:
                self._slide_os = openslide.OpenSlide(str(self.path))
                self.backend = "openslide"
                return
            except Exception:
                pass
        
        # Fallback a tifffile
        if tifffile is not None:
            try:
                self._slide_tf = tifffile.TiffFile(str(self.path))
                self.backend = "tifffile"
                return
            except Exception:
                pass

        raise ValueError(f"No se pudo abrir la imagen {self.path}. Formato no soportado o dependencias faltantes (openslide, tifffile).")

    @property
    def level_count(self) -> int:
        if self.backend == "openslide":
            return self._slide_os.level_count
        elif self.backend == "tifffile":
            return len(self._slide_tf.series[0].levels)
        return 1

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Dimensiones del nivel 0 (ancho, alto)."""
        if self.backend == "openslide":
            return self._slide_os.dimensions
        elif self.backend == "tifffile":
            shape = self._slide_tf.series[0].shape
            # (Y, X, C) o (C, Y, X)
            if len(shape) >= 2:
                # Tifffile usualmente retorna (alto, ancho) o (alto, ancho, canales)
                # Queremos (ancho, alto)
                if shape[-1] in (1, 3, 4): 
                    return (shape[1], shape[0])
                else:
                    return (shape[1], shape[0])
        return (0, 0)

    @property
    def mpp(self) -> Tuple[Optional[float], Optional[float]]:
        """Resolución física en micrones por píxel (mpp_x, mpp_y)."""
        if self.backend == "openslide":
            mpp_x = self._slide_os.properties.get(openslide.PROPERTY_NAME_MPP_X)
            mpp_y = self._slide_os.properties.get(openslide.PROPERTY_NAME_MPP_Y)
            return (float(mpp_x) if mpp_x else None, float(mpp_y) if mpp_y else None)
        elif self.backend == "tifffile":
            try:
                page = self._slide_tf.pages[0]
                res_x = page.tags['XResolution'].value
                res_y = page.tags['YResolution'].value
                res_unit = page.tags['ResolutionUnit'].value
                # Convertir a micrones si es necesario
                # Si res_unit == 3 (centímetro), 10000 / res
                mpp_x = 10000 / res_x[0] * res_x[1] if res_unit == 3 else None
                mpp_y = 10000 / res_y[0] * res_y[1] if res_unit == 3 else None
                return mpp_x, mpp_y
            except KeyError:
                return None, None
        return None, None

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> Image.Image:
        """
        Lee una región de la WSI.
        location: (x, y) en el nivel de máxima resolución (nivel 0).
        level: nivel de resolución a leer.
        size: (ancho, alto) de la región a extraer.
        """
        if self.backend == "openslide":
            img = self._slide_os.read_region(location, level, size)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        elif self.backend == "tifffile":
            # Para tifffile sin backend avanzado de pirámide (Zarr/Dask),
            # leemos el nivel correspondiente. Requiere ajuste manual de coordenadas.
            # Implementación simplificada (asume carga en RAM si es pequeña o uso de asarray)
            series = self._slide_tf.series[0]
            level_page = series.levels[level]
            downsample = series.shape[0] / level_page.shape[0]
            
            x, y = location
            w, h = size
            # Coordenadas escaladas
            xs = int(x / downsample)
            ys = int(y / downsample)
            
            data = level_page.asarray()
            # Asumimos (H, W, C)
            patch_data = data[ys:ys+h, xs:xs+w]
            return Image.fromarray(patch_data).convert("RGB")

        raise RuntimeError("Backend no inicializado.")

    def close(self):
        if self._slide_os:
            self._slide_os.close()
        if self._slide_tf:
            self._slide_tf.close()
