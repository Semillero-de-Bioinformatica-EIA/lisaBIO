import pandas as pd
from pathlib import Path
from typing import Optional, Dict

class METABRICLoader:
    """
    Carga de datos de METABRIC.
    Requiere que las rutas se provean explícitamente y no usa rutas hardcodeadas.
    No genera datos sintéticos silenciosamente.
    """
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"El directorio de METABRIC no existe: {self.data_dir}")

    def load_clinical(self) -> pd.DataFrame:
        path = self.data_dir / "data_clinical_patient.txt"
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        return pd.read_csv(path, sep="\t", comment="#").set_index("PATIENT_ID")

    def load_mrna(self) -> pd.DataFrame:
        path = self.data_dir / "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        df = pd.read_csv(path, sep="\t", comment="#")
        if "Hugo_Symbol" in df.columns:
            df = df.set_index("Hugo_Symbol")
            # Eliminar columnas de metadatos si existen
            cols_to_drop = ["Entrez_Gene_Id"]
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            return df.T
        return df

    def load_cna(self) -> pd.DataFrame:
        path = self.data_dir / "data_cna.txt"
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        df = pd.read_csv(path, sep="\t", comment="#")
        if "Hugo_Symbol" in df.columns:
            df = df.set_index("Hugo_Symbol")
            cols_to_drop = ["Entrez_Gene_Id"]
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            return df.T
        return df

    def load_mutations(self) -> pd.DataFrame:
        path = self.data_dir / "data_mutations.txt"
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        # Retorna raw data, el adaptador se encargará de pivotear
        return pd.read_csv(path, sep="\t", comment="#")
