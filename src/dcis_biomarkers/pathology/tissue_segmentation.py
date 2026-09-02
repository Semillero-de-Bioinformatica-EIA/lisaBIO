import numpy as np

def segment_tissue(rgb_image: np.ndarray, threshold: int = 210) -> np.ndarray:
    """
    Segmentación automática de áreas de tejido tumoral/estromal excluyendo fondo blanco en la tinción H&E.
    Retorna una máscara binaria (1: tejido, 0: fondo).
    """
    # Convertir a escala de grises para filtrado HSV / Luminosidad
    gray = np.mean(rgb_image, axis=2)
    tissue_mask = (gray < threshold).astype(np.uint8)
    return tissue_mask
