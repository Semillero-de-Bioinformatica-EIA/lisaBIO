import pandas as pd
import numpy as np
from typing import List, Optional
from .base import ModalityAdapter

class ClinicalAdapter(ModalityAdapter):
    """
    Adaptador para variables clínicas.
    Realiza imputación (mediana para numéricas, moda para categóricas) y codificación.
    """
    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        super().__init__()
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        self.num_imputers = {}
        self.cat_imputers = {}
        self.cat_encoders = {}
        
    def fit(self, df: pd.DataFrame) -> 'ClinicalAdapter':
        # Aprender parámetros de numéricas
        for col in self.num_cols:
            if col in df.columns:
                self.num_imputers[col] = df[col].median()
            else:
                self.num_imputers[col] = 0.0
                
        # Aprender parámetros de categóricas
        for col in self.cat_cols:
            if col in df.columns:
                self.cat_imputers[col] = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                # Crear variables dummy
                unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
                self.cat_encoders[col] = unique_vals
            else:
                self.cat_imputers[col] = "Unknown"
                self.cat_encoders[col] = []
                
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.num_imputers and not self.cat_imputers:
            raise RuntimeError("El adaptador clínico no ha sido fiteado.")
            
        df_out = pd.DataFrame(index=df.index)
        
        # Numéricas
        for col in self.num_cols:
            if col in df.columns:
                # Imputar y asignar
                df_out[col] = pd.to_numeric(df[col], errors='coerce').fillna(self.num_imputers[col])
            else:
                df_out[col] = self.num_imputers[col]
                
        # Categóricas (One-Hot Encoding)
        for col in self.cat_cols:
            if col in df.columns:
                s = df[col].fillna(self.cat_imputers[col]).astype(str)
            else:
                s = pd.Series(self.cat_imputers[col], index=df.index)
                
            for val in self.cat_encoders[col]:
                col_name = f"{col}_{val}"
                df_out[col_name] = (s == val).astype(float)
                
        return df_out
