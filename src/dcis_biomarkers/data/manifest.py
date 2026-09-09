import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

class CaseManifest:
    """
    Manifiesto central de casos. Garantiza el contrato de datos alineando
    pacientes, muestras, bloques, láminas y modalidades.
    """
    EXPECTED_COLUMNS = [
        "patient_id",
        "sample_id",
        "specimen_id",
        "block_id",
        "slide_id",
        "center_id",
        "acquisition_date",
        "modality",
        "file_path",
        "label",
        "time_to_event",
        "event_observed",
        "split",
        "quality_status"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._validate()

    @classmethod
    def load_parquet(cls, path: str | Path) -> "CaseManifest":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest no encontrado en {path}")
        return cls(pd.read_parquet(path))

    def _validate(self):
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en el manifiesto: {missing_cols}")
        
        # Validación de duplicados (misma modalidad, misma slide, mismo paciente)
        # Esto depende de las reglas exactas, por ahora solo verificamos que no haya
        # filas completamente idénticas
        if self.df.duplicated().any():
            raise ValueError("El manifiesto contiene filas duplicadas exactas.")

    def get_patient_data(self, patient_id: str) -> pd.DataFrame:
        return self.df[self.df["patient_id"] == patient_id]


class FeatureManifest:
    """
    Manifiesto de características para trazar genes, sondas y otras variables.
    """
    EXPECTED_COLUMNS = [
        "feature_id",
        "feature_name",
        "modality",
        "gene_id",
        "unit",
        "transformation",
        "batch",
        "reference_genome",
        "missing_fraction",
        "quality_status"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._validate()

    @classmethod
    def load_parquet(cls, path: str | Path) -> "FeatureManifest":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest no encontrado en {path}")
        return cls(pd.read_parquet(path))

    def _validate(self):
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en el manifiesto de features: {missing_cols}")

        if self.df["feature_id"].duplicated().any():
            raise ValueError("El manifiesto de features contiene feature_ids duplicados.")


def validate_manifest(case_manifest_path: str | Path, feature_manifest_path: str | Path, output_report_path: str | Path):
    """
    Genera un reporte de validación del manifiesto.
    """
    report: Dict[str, Any] = {"status": "success", "errors": [], "warnings": [], "stats": {}}
    
    try:
        case_manifest = CaseManifest.load_parquet(case_manifest_path)
        report["stats"]["total_cases"] = len(case_manifest.df)
        report["stats"]["unique_patients"] = case_manifest.df["patient_id"].nunique()
        report["stats"]["modalities"] = case_manifest.df["modality"].unique().tolist()
    except Exception as e:
        report["status"] = "failed"
        report["errors"].append(f"Error en CaseManifest: {e}")

    try:
        feature_manifest = FeatureManifest.load_parquet(feature_manifest_path)
        report["stats"]["total_features"] = len(feature_manifest.df)
    except Exception as e:
        report["status"] = "failed"
        report["errors"].append(f"Error en FeatureManifest: {e}")

    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    return report
