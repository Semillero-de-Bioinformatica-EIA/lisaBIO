import pytest
import pandas as pd
from dcis_biomarkers.data.manifest import CaseManifest, FeatureManifest

def test_case_manifest_validation():
    # Faltan columnas
    df_invalid = pd.DataFrame({"patient_id": ["P1", "P2"]})
    with pytest.raises(ValueError):
        CaseManifest(df_invalid)
        
    # Correcto
    df_valid = pd.DataFrame({col: [] for col in CaseManifest.EXPECTED_COLUMNS})
    manifest = CaseManifest(df_valid)
    assert len(manifest.df) == 0

def test_case_manifest_duplicates():
    df_valid = pd.DataFrame({col: ["A", "A"] for col in CaseManifest.EXPECTED_COLUMNS})
    with pytest.raises(ValueError, match="El manifiesto contiene filas duplicadas exactas"):
        CaseManifest(df_valid)

def test_feature_manifest_validation():
    df_invalid = pd.DataFrame({"feature_id": ["F1"]})
    with pytest.raises(ValueError):
        FeatureManifest(df_invalid)

def test_feature_manifest_duplicates():
    df_valid = pd.DataFrame({col: ["F1", "F1"] for col in FeatureManifest.EXPECTED_COLUMNS})
    with pytest.raises(ValueError, match="feature_ids duplicados"):
        FeatureManifest(df_valid)
