import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import StratifiedKFold

class PatientSplitter:
    """
    Asegura división a nivel de paciente para prevenir fuga de datos.
    """
    def __init__(self, df: pd.DataFrame, patient_col: str = "patient_id", label_col: str = "label"):
        self.df = df
        self.patient_col = patient_col
        self.label_col = label_col

    def split_kfold(self, n_splits: int = 5, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Retorna listas de DataFrames (train, test) estratificados por paciente.
        """
        # Obtenemos un df único por paciente para estratificar
        # Asumimos que la etiqueta de un paciente no cambia
        patient_df = self.df.drop_duplicates(subset=[self.patient_col]).copy()
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []
        
        for train_idx, test_idx in skf.split(patient_df, patient_df[self.label_col]):
            train_patients = patient_df.iloc[train_idx][self.patient_col].values
            test_patients = patient_df.iloc[test_idx][self.patient_col].values
            
            train_df = self.df[self.df[self.patient_col].isin(train_patients)].copy()
            test_df = self.df[self.df[self.patient_col].isin(test_patients)].copy()
            
            assert_no_leakage(train_df, test_df, self.patient_col)
            splits.append((train_df, test_df))
            
        return splits

def assert_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, patient_col: str = "patient_id"):
    """
    Verifica que no exista solapamiento de pacientes entre train y test.
    """
    train_patients = set(train_df[patient_col].unique())
    test_patients = set(test_df[patient_col].unique())
    intersection = train_patients.intersection(test_patients)
    
    if intersection:
        raise ValueError(f"¡FUGA DE DATOS DETECTADA! Pacientes en train y test: {intersection}")
