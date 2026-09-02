import pytest
import pandas as pd
import numpy as np
from dcis_biomarkers.multiomics import OmicsPreprocessor, run_differential_expression

def test_omics_preprocessor():
    df_dummy = pd.DataFrame(
        np.random.randint(0, 100, size=(10, 50)),
        index=[f"sample_{i}" for i in range(10)],
        columns=[f"gene_{j}" for j in range(50)]
    )
    preprocessor = OmicsPreprocessor(min_counts=5, min_genes=10)
    processed = preprocessor.fit_transform(df_dummy)
    assert not processed.empty
    assert processed.shape[0] <= 10

def test_differential_expression():
    df_expr = pd.DataFrame(
        np.random.rand(10, 20),
        index=[f"s_{i}" for i in range(10)],
        columns=[f"g_{j}" for j in range(20)]
    )
    labels = pd.Series(["Invasive"]*5 + ["DCIS_Indolent"]*5, index=df_expr.index)
    res = run_differential_expression(df_expr, labels)
    assert "log2FC" in res.columns
    assert "padj" in res.columns
