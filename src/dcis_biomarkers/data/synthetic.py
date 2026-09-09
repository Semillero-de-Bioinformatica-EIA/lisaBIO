import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_paired_dataset(output_dir: str | Path, num_patients: int = 20):
    """
    Genera un dataset sintético con datos pareados (clínicos, ómicos e imágenes dummy)
    para pruebas técnicas end-to-end.
    
    ¡MARCADO EXPLÍCITAMENTE COMO SINTÉTICO!
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    patients = [f"SYNTH_PAT_{i:03d}" for i in range(num_patients)]
    labels = np.random.randint(0, 3, size=num_patients)
    
    # Manifest
    manifest_records = []
    for i, p in enumerate(patients):
        manifest_records.append({
            "patient_id": p,
            "sample_id": f"SAMP_{p}",
            "specimen_id": f"SPEC_{p}",
            "block_id": f"BLK_{p}",
            "slide_id": f"SLD_{p}",
            "center_id": "SYNTH_CENTER",
            "acquisition_date": "2024-01-01",
            "modality": "WSI",
            "file_path": f"synthetic/{p}.tif", # Dummy path
            "label": labels[i],
            "time_to_event": np.random.exponential(50),
            "event_observed": np.random.binomial(1, 0.3),
            "split": "train" if i < int(num_patients*0.8) else "test",
            "quality_status": "PASS"
        })
    
    df_manifest = pd.DataFrame(manifest_records)
    manifest_path = output_dir / "cases.parquet"
    df_manifest.to_parquet(manifest_path)
    
    # Feature Manifest (Omics)
    num_features = 100
    features = [f"SYNTH_GENE_{j:03d}" for j in range(num_features)]
    df_features = pd.DataFrame({
        "feature_id": features,
        "feature_name": features,
        "modality": "Transcriptomics",
        "gene_id": features,
        "unit": "log2(CPM)",
        "transformation": "log1p",
        "batch": "SYNTH_BATCH",
        "reference_genome": "GRCh38",
        "missing_fraction": 0.0,
        "quality_status": "PASS"
    })
    feature_manifest_path = output_dir / "feature_manifest.parquet"
    df_features.to_parquet(feature_manifest_path)
    
    # Omics Data (Dummy)
    omics_data = np.random.randn(num_patients, num_features)
    df_omics = pd.DataFrame(omics_data, index=patients, columns=features)
    df_omics.index.name = "patient_id"
    omics_path = output_dir / "omics_data.parquet"
    df_omics.to_parquet(omics_path)
    
    # Escribir archivo de advertencia
    warning_path = output_dir / "SYNTHETIC_DATA_WARNING.txt"
    with open(warning_path, "w") as f:
        f.write("ESTOS DATOS SON SINTÉTICOS Y SOLO DEBEN USARSE PARA PRUEBAS TÉCNICAS DEL PIPELINE.\n")
        f.write("NO TIENEN VALIDEZ BIOLÓGICA NI CLÍNICA.\n")
        
    return manifest_path, feature_manifest_path, omics_path
