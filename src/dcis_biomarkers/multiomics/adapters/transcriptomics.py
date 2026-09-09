import pandas as pd
import numpy as np
from typing import List, Optional
from .base import ModalityAdapter

class TranscriptomicsAdapter(ModalityAdapter):
    """
    Adaptador para transcriptómica (RNA-seq / Microarray).
    Realiza filtrado por varianza y normalización Z-score.
    """
    def __init__(self, top_n_hvg: int = 1000):
        super().__init__()
        self.top_n_hvg = top_n_hvg
        self.selected_genes: Optional[List[str]] = None
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> 'TranscriptomicsAdapter':
        # 1. Seleccionar Highly Variable Genes (HVG)
        variances = df.var()
        self.selected_genes = variances.nlargest(self.top_n_hvg).index.tolist()
        
        # 2. Calcular mean y std en el subset de HVG
        df_subset = df[self.selected_genes]
        self.means = df_subset.mean()
        self.stds = df_subset.std()
        
        # Evitar división por cero
        self.stds[self.stds < 1e-6] = 1e-6
        
        self.qc_metrics["n_genes_selected"] = len(self.selected_genes)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.selected_genes is None or self.means is None or self.stds is None:
            raise RuntimeError("El adaptador debe ser fiteado antes de transform().")
            
        # Filtrar a los genes seleccionados en train
        df_subset = df.reindex(columns=self.selected_genes, fill_value=0.0)
        
        # Normalizar Z-score con parámetros de train
        df_norm = (df_subset - self.means) / self.stds
        return df_norm
