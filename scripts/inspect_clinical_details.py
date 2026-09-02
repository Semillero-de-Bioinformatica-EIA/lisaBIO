"""
Script de detalle clínico para METABRIC.
Examina las columnas clínicas disponibles para supervivencia, subtipos de cáncer de mama y tratamiento.
"""

from pathlib import Path

METABRIC_DIR = Path(r"C:\Users\loapi\Downloads\brca_metabric\brca_metabric")


def inspect_clinical_cols():
    patient_file = METABRIC_DIR / "data_clinical_patient.txt"
    sample_file = METABRIC_DIR / "data_clinical_sample.txt"

    print("=== PATIENT CLINICAL DATA (data_clinical_patient.txt) ===")
    with open(patient_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split('\t')
            print(f"Total columnas: {len(cols)}")
            for i, col in enumerate(cols):
                print(f"  {i+1:2d}. {col}")
            break

    print("\n=== SAMPLE CLINICAL DATA (data_clinical_sample.txt) ===")
    with open(sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split('\t')
            print(f"Total columnas: {len(cols)}")
            for i, col in enumerate(cols):
                print(f"  {i+1:2d}. {col}")
            break


if __name__ == "__main__":
    inspect_clinical_cols()
