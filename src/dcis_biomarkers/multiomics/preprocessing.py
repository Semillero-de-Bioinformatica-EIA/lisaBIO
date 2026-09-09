import pandas as pd
import numpy as np
from typing import Dict, Any

from .adapters.base import ModalityAdapter
from .adapters.transcriptomics import TranscriptomicsAdapter
from .adapters.clinical import ClinicalAdapter

class OmicsPreprocessor:
    """
    Gestor de preprocesamiento para múltiples modalidades ómicas.
    Coordina los adaptadores de cada modalidad y previene fuga de datos.
    """
    def __init__(self, adapters: Dict[str, ModalityAdapter]):
        self.adapters = adapters
        self.is_fitted = False

    def fit(self, data_dict: Dict[str, pd.DataFrame]) -> 'OmicsPreprocessor':
        """
        Ajusta todos los adaptadores usando SOLO datos de entrenamiento.
        data_dict: { "transcriptomics": df_rna, "clinical": df_clin, ... }
        """
        for mod, df in data_dict.items():
            if mod in self.adapters:
                self.adapters[mod].fit(df)
        self.is_fitted = True
        return self

    def transform(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transforma los datos usando los parámetros aprendidos en fit().
        """
        if not self.is_fitted:
            raise RuntimeError("Debe llamar a fit() antes de transform()")
            
        out_dict = {}
        for mod, df in data_dict.items():
            if mod in self.adapters:
                out_dict[mod] = self.adapters[mod].transform(df)
        return out_dict

    def fit_transform(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return self.fit(data_dict).transform(data_dict)
