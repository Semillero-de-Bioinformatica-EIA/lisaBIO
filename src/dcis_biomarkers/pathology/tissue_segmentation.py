import numpy as np
import cv2
from typing import Dict, Tuple

def segment_tissue(image_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Segmenta el tejido de H&E usando conversión a HSV y Otsu thresholding.
    Implementación mejorada con operaciones morfológicas y métricas QC.
    
    Args:
        image_rgb: Numpy array (H, W, 3) en RGB.
        
    Returns:
        Tuple[mascara_binaria (H, W), metricas_qc]
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Extraer el canal de saturación (S)
    s_channel = hsv[:, :, 1]
    
    # Aplicar umbral de Otsu en el canal de saturación
    # El tejido H&E (rosado/púrpura) tiende a tener mayor saturación que el fondo blanco.
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas para limpiar ruido y cerrar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    mask_bin = (mask > 0).astype(np.uint8)
    
    # Calcular métricas QC
    total_pixels = mask_bin.size
    tissue_pixels = np.sum(mask_bin)
    tissue_percentage = (tissue_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    metrics = {
        "tissue_percentage": tissue_percentage,
        "total_pixels": int(total_pixels),
        "tissue_pixels": int(tissue_pixels)
    }
    
    return mask_bin, metrics
