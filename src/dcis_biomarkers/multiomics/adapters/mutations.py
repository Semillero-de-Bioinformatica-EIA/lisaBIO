import pandas as pd
from typing import List, Optional
from .base import ModalityAdapter

class MutationAdapter(ModalityAdapter):
    """
    Adaptador para mutaciones somáticas.
    Representa explícitamente: absent (0), unmeasured (-1), observed (1).
    """
    def __init__(self, top_n_genes: int = 500):
        super().__init__()
        self.top_n_genes = top_n_genes
        self.selected_genes: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> 'MutationAdapter':
        # Seleccionar genes mutados más frecuentemente
        freq = (df > 0).sum()
        self.selected_genes = freq.nlargest(self.top_n_genes).index.tolist()
        self.qc_metrics["n_genes_selected"] = len(self.selected_genes)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.selected_genes is None:
            raise RuntimeError("El adaptador debe ser fiteado.")
            
        df_subset = df.reindex(columns=self.selected_genes, fill_value=-1)
        
        # Binarizar (0, 1) y mantener -1 para missing
        # Asumiendo que >0 es mutación observada
        mask_measured = df_subset != -1
        df_out = df_subset.copy()
        df_out[mask_measured] = (df_out[mask_measured] > 0).astype(float)
        
        return df_out
