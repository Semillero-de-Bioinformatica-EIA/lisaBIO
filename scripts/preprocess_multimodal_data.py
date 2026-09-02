"""
Preprocesamiento e Integracion Multimodal de Datos.
Genera indices procesados y verifica la integridad de parches TIF MPM/H&E y tablas de METABRIC.
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.pathology.mpm_dataset import MPMSequenceDataset
from dcis_biomarkers.multiomics.metabric_loader import METABRICDataset


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"


def preprocess_all_datasets():
    print("=== INICIANDO PREPROCESAMIENTO Y VERIFICACION MULTIMODAL ===")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Indexar dataset de Patologia MPM TIF / H&E
    print("\n1. Procesando Dataset de Patologia MPM (PKG - HE-vs-MPM)...")
    mpm_dataset = MPMSequenceDataset(is_training=False)
    print(f"   |-- Total de casos de patologia procesados: {len(mpm_dataset)}")
    
    records = []
    for idx in range(len(mpm_dataset)):
        item = mpm_dataset[idx]
        records.append({
            "case_id": item["case_id"],
            "seq_len": item["seq_len"],
            "diagnosis": item["diagnosis"],
            "label": item["label"].item()
        })
    df_mpm = pd.DataFrame(records)
    mpm_summary_path = PROCESSED_DIR / "mpm_processed_summary.csv"
    df_mpm.to_csv(mpm_summary_path, index=False)
    print(f"   +-- Resumen guardado en: {mpm_summary_path}")

    # 2. Indexar dataset de Genomica METABRIC
    print("\n2. Procesando Dataset Omico METABRIC (Expresion Z-Scores & Clinica)...")
    metabric_dataset = METABRICDataset(num_top_genes=1000, is_training=False)
    print(f"   |-- Total de pacientes METABRIC cargados: {len(metabric_dataset)}")
    
    metabric_summary_path = PROCESSED_DIR / "metabric_processed_summary.csv"
    df_meta = pd.DataFrame({
        "patient_id": metabric_dataset.patient_ids,
        "rfs_time": metabric_dataset.survival_targets[:, 0],
        "rfs_event": metabric_dataset.survival_targets[:, 1]
    })
    df_meta.to_csv(metabric_summary_path, index=False)
    print(f"   +-- Resumen guardado en: {metabric_summary_path}")

    print("\n=== PREPROCESAMIENTO MULTIMODAL COMPLETADO CON EXITO ===")


if __name__ == "__main__":
    preprocess_all_datasets()
