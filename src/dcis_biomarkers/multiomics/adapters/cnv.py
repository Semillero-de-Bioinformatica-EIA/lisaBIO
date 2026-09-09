import pandas as pd
from typing import List, Optional
from .base import ModalityAdapter

class CNVAdapter(ModalityAdapter):
    """
    Adaptador para Copy Number Variations.
    """
    def __init__(self, top_n_genes: int = 500):
        super().__init__()
        self.top_n_genes = top_n_genes
        self.selected_genes: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> 'CNVAdapter':
        # Seleccionar genes con mayor varianza
        var = df.var()
        self.selected_genes = var.nlargest(self.top_n_genes).index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.selected_genes is None:
            raise RuntimeError("El adaptador debe ser fiteado.")
        
        # CNV típicamente ya viene normalizado o discreto (e.g., -2, -1, 0, 1, 2)
        df_subset = df.reindex(columns=self.selected_genes, fill_value=0.0)
        return df_subset
