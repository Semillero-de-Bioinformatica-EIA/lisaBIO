"""
Dataset Ingestion and Indexing Script for PKG - HE-vs-MPM.
Scans the dataset directory, extracts case classes (DCIS, DCISM, IDC),
indexes H&E, CK56, and MPM images, and exports a structured dataset index CSV.
"""

import os
import glob
import re
import pandas as pd
from pathlib import Path

DATASET_ROOT = r"C:\Users\loapi\Downloads\PKG - HE-vs-MPM"
OUTPUT_CSV = r"data/processed/he_mpm_dataset_index.csv"


def parse_case_info(case_name: str):
    """
    Extrae el número de caso y la categoría diagnóstica/estadio de progresión.
    Ejemplo: 'Case1-DCISM' -> Case=1, Class='DCISM'
    """
    match = re.match(r"Case(\d+)-(DCIS|DCISM|IDC)", case_name)
    if match:
        case_num = int(match.group(1))
        diagnosis = match.group(2)
        return case_num, diagnosis
    return None, "Unknown"


def scan_he_mpm_dataset(dataset_root: str = DATASET_ROOT) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    he_dir = dataset_root / "H&E-stained"
    ck_dir = dataset_root / "CK56-stained"
    mpm_dir = dataset_root / "MPM image"

    records = []

    # 1. Escanear Casos H&E y CK56
    if he_dir.exists():
        for case_folder in sorted(he_dir.glob("Case*")):
            if case_folder.is_dir():
                case_name = case_folder.name
                case_num, diagnosis = parse_case_info(case_name)
                
                # Archivos H&E
                he_svs = list(case_folder.glob("*.svs")) + list(case_folder.glob("*.svs.partial"))
                he_xml = list(case_folder.glob("*.xml")) + list(case_folder.glob("*.xml.partial"))
                
                # Archivos CK56
                ck_folder = ck_dir / case_name
                ck_svs = list(ck_folder.glob("*.svs")) + list(ck_folder.glob("*.svs.partial")) if ck_folder.exists() else []

                records.append({
                    "case_id": case_name,
                    "case_number": case_num,
                    "diagnosis": diagnosis,
                    "modality": "Whole_Slide",
                    "he_svs_path": str(he_svs[0]) if he_svs else None,
                    "he_xml_path": str(he_xml[0]) if he_xml else None,
                    "ck56_svs_path": str(ck_svs[0]) if ck_svs else None,
                    "mpm_tif_path": None,
                    "roi_id": None
                })

    # 2. Escanear Imágenes MPM (Multi-Photon Microscopy)
    if mpm_dir.exists():
        mpm_files = sorted(mpm_dir.glob("Case*.tif"))
        for mpm_file in mpm_files:
            file_name = mpm_file.name
            match = re.match(r"Case(\d+)-ROI-(\d+)\.tif", file_name)
            if match:
                case_num = int(match.group(1))
                roi_num = int(match.group(2))
                
                # Mapear diagnóstico del caso
                diag_map = {11: "DCIS", 12: "IDC"}
                diagnosis = diag_map.get(case_num, "DCISM")
                
                records.append({
                    "case_id": f"Case{case_num}-{diagnosis}",
                    "case_number": case_num,
                    "diagnosis": diagnosis,
                    "modality": "MPM_Microscopy",
                    "he_svs_path": None,
                    "he_xml_path": None,
                    "ck56_svs_path": None,
                    "mpm_tif_path": str(mpm_file),
                    "roi_id": f"ROI-{roi_num}"
                })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print(f"Escanando dataset en: {DATASET_ROOT}")
    df = scan_he_mpm_dataset()
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n--- RESUMEN DEL DATASET ---")
    print(f"Total de registros indexados: {len(df)}")
    print("\nDistribución por Modalidad:")
    print(df["modality"].value_counts())
    print("\nDistribución por Diagnóstico:")
    print(df.groupby(["diagnosis", "modality"]).size())
    print(f"\nÍndice guardado con éxito en: {OUTPUT_CSV}")
