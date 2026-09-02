import numpy as np
from typing import List, Tuple, Dict, Any

def extract_wsi_attention_heatmap(coordinates: List[Tuple[int, int]], attention_weights: np.ndarray, patch_size: int = 256) -> Dict[str, Any]:
    """
    Mapea las ponderaciones de atención generadas por el modelo de IA hacia las coordenadas
    originales de la imagen de lámina completa (WSI) para visualizar regiones tisulares críticas.
    """
    heatmap_records = []
    norm_weights = (attention_weights - np.min(attention_weights)) / (np.max(attention_weights) - np.min(attention_weights) + 1e-8)

    for (x, y), weight in zip(coordinates, norm_weights):
        heatmap_records.append({
            'x': x,
            'y': y,
            'patch_size': patch_size,
            'attention_score': float(weight)
        })

    return {
        "num_patches": len(coordinates),
        "heatmap": heatmap_records
    }
