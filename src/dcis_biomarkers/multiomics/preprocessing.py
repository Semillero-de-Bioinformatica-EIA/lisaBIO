import numpy as np
import pandas as pd
from typing import Tuple

class OmicsPreprocessor:
    """Clase para filtrado, normalización y transformación de matrices transcriptómicas y genómicas."""

    def __init__(self, min_counts: int = 10, min_genes: int = 200, log_transform: bool = True):
        self.min_counts = min_counts
        self.min_genes = min_genes
        self.log_transform = log_transform

    def filter_low_expression(self, df_counts: pd.DataFrame) -> pd.DataFrame:
        """Filtra genes con conteos bajos y muestras con pocos genes expresados."""
        # Filtrar muestras (filas)
        valid_samples = (df_counts > 0).sum(axis=1) >= self.min_genes
        filtered_df = df_counts.loc[valid_samples]

        # Filtrar genes (columnas)
        valid_genes = filtered_df.sum(axis=0) >= self.min_counts
        return filtered_df.loc[:, valid_genes]

    def normalize_cpm(self, df_counts: pd.DataFrame) -> pd.DataFrame:
        """Calcula Counts Per Million (CPM) y aplica transformación log1p."""
        lib_sizes = df_counts.sum(axis=1)
        cpm = df_counts.div(lib_sizes, axis=0) * 1e6
        if self.log_transform:
            return np.log1p(cpm)
        return cpm

    def fit_transform(self, df_counts: pd.DataFrame) -> pd.DataFrame:
        """Pipeline completo de filtrado y normalización."""
        filtered = self.filter_low_expression(df_counts)
        normalized = self.normalize_cpm(filtered)
        return normalized
