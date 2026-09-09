import pytest
from pathlib import Path
import tempfile
import sys
import subprocess

def test_synthetic_data_generation_and_validation():
    """Prueba de integración end-to-end usando datos sintéticos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from dcis_biomarkers.data.synthetic import generate_synthetic_paired_dataset
        from dcis_biomarkers.data.manifest import validate_manifest
        
        # Generar
        manifest_path, feat_path, omics_path = generate_synthetic_paired_dataset(tmpdir, num_patients=5)
        
        # Validar
        report_path = Path(tmpdir) / "report.json"
        report = validate_manifest(manifest_path, feat_path, report_path)
        
        assert report["status"] == "success"
        assert report["stats"]["total_cases"] == 5
        assert report["stats"]["total_features"] == 100
        
        # Opcional: Probar que la advertencia existe
        warning_path = Path(tmpdir) / "SYNTHETIC_DATA_WARNING.txt"
        assert warning_path.exists()
