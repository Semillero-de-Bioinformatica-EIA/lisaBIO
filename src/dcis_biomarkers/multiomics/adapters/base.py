import pandas as pd
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

class ModalityAdapter(ABC):
    """
    Clase base para adaptadores de modalidades multi-ómicas.
    Estandariza la carga, validación y control de calidad.
    """
    def __init__(self):
        self.qc_metrics: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'ModalityAdapter':
        """Aprende parámetros de normalización (solo en train)."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformación a los datos."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_qc_metrics(self) -> Dict[str, Any]:
        return self.qc_metrics
