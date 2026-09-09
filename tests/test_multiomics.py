import pytest
import pandas as pd
import numpy as np

from dcis_biomarkers.multiomics.adapters.transcriptomics import TranscriptomicsAdapter
from dcis_biomarkers.multiomics.adapters.clinical import ClinicalAdapter
from dcis_biomarkers.multiomics.preprocessing import OmicsPreprocessor

def test_transcriptomics_adapter():
    df_train = pd.DataFrame(np.random.randn(10, 50), columns=[f"gene_{i}" for i in range(50)])
    df_test = pd.DataFrame(np.random.randn(5, 50), columns=[f"gene_{i}" for i in range(50)])
    
    adapter = TranscriptomicsAdapter(top_n_hvg=10)
    adapter.fit(df_train)
    
    out_train = adapter.transform(df_train)
    out_test = adapter.transform(df_test)
    
    assert out_train.shape == (10, 10)
    assert out_test.shape == (5, 10)
    assert len(adapter.selected_genes) == 10

def test_clinical_adapter():
    df_train = pd.DataFrame({
        "age": [40, 50, None, 60],
        "grade": ["1", "2", "2", None]
    })
    
    df_test = pd.DataFrame({
        "age": [45, None],
        "grade": ["3", "1"]
    })
    
    adapter = ClinicalAdapter(num_cols=["age"], cat_cols=["grade"])
    adapter.fit(df_train)
    
    out_train = adapter.transform(df_train)
    out_test = adapter.transform(df_test)
    
    # 1 col num, 2 dummy cols for grade (1, 2) -> 3 cols
    assert "age" in out_train.columns
    assert "grade_1" in out_train.columns
    assert "grade_2" in out_train.columns
    
    # grade_3 no estaba en train, test debe tener todo ceros en grade_1 y grade_2
    assert out_test.loc[0, "grade_1"] == 0.0
    assert out_test.loc[0, "grade_2"] == 0.0

def test_omics_preprocessor():
    df_train = pd.DataFrame(np.random.randn(10, 50), columns=[f"gene_{i}" for i in range(50)])
    adapters = {
        "transcriptomics": TranscriptomicsAdapter(top_n_hvg=10)
    }
    prep = OmicsPreprocessor(adapters)
    out = prep.fit_transform({"transcriptomics": df_train})
    assert "transcriptomics" in out
    assert out["transcriptomics"].shape == (10, 10)
