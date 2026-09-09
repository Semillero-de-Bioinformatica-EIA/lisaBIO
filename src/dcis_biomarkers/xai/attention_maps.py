from typing import Dict, Any, List
import pandas as pd
from pathlib import Path

def generate_attention_maps(attention_weights: List[float], metadata: Dict[str, Any], output_path: str | Path):
    """
    Genera y guarda mapas de atención de regiones WSI con metadatos completos.
    
    metadata debe incluir: patient_id, slide_id, coordinates, level, physical_size, checksum, model_version
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # En lugar de generar imágenes directamente (lo cual depende de CV2/PIL complejo aquí),
    # guardamos un reporte detallado que puede ser renderizado posteriormente por un script de UI.
    
    df = pd.DataFrame({
        "roi_index": range(len(attention_weights)),
        "attention_score": attention_weights
    })
    
    # Adjuntamos metadata como un archivo JSON separado o integrarlo en la base de datos de atención.
    import json
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    df.to_csv(output_path, index=False)
    
    return str(output_path)
