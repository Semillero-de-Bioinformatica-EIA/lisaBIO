import pytest
import pandas as pd
from dcis_biomarkers.data.splits import PatientSplitter, assert_no_leakage

def test_no_leakage_assertion():
    train_df = pd.DataFrame({"patient_id": ["P1", "P2", "P3"]})
    test_df_clean = pd.DataFrame({"patient_id": ["P4", "P5"]})
    test_df_leaky = pd.DataFrame({"patient_id": ["P3", "P6"]}) # P3 is in both
    
    # Clean split should pass
    assert_no_leakage(train_df, test_df_clean)
    
    # Leaky split should raise ValueError
    with pytest.raises(ValueError, match="FUGA DE DATOS DETECTADA"):
        assert_no_leakage(train_df, test_df_leaky)

def test_patient_splitter():
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P3", "P4", "P4", "P5", "P6"],
        "label": [0, 0, 1, 0, 1, 1, 0, 1]
    })
    
    splitter = PatientSplitter(df)
    splits = splitter.split_kfold(n_splits=2, random_state=42)
    
    assert len(splits) == 2
    for train_df, test_df in splits:
        assert_no_leakage(train_df, test_df)
        assert len(train_df) + len(test_df) == len(df)
