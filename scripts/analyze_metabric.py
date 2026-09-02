"""
METABRIC Dataset Inspection and Functional Data Analysis Script.
Analiza la integridad, dimensiones, muestras y características funcionales
del dataset BRCA METABRIC (Expresión génica, clínica, mutaciones, CNA, metilación).
"""

import os
from pathlib import Path

METABRIC_DIR = Path(r"C:\Users\loapi\Downloads\brca_metabric\brca_metabric")


def inspect_tsv_header_and_shape(file_path: Path, max_rows: int = 5):
    """
    Inspecciona encabezados, comentarios (comenzando con #) y número total de filas/columnas.
    """
    if not file_path.exists():
        return None

    size_mb = file_path.stat().st_size / (1024 * 1024)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = []
        for line in f:
            if not line.startswith('#'):
                lines.append(line.strip().split('\t'))
            if len(lines) >= max_rows:
                break
                
    # Contar total de líneas aproximadamente
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        total_lines = sum(1 for line in f if not line.startswith('#'))

    num_cols = len(lines[0]) if lines else 0
    header = lines[0] if lines else []
    
    return {
        "file": file_path.name,
        "size_mb": round(size_mb, 2),
        "total_rows": total_lines - 1 if total_lines > 0 else 0, # Excluyendo header
        "total_cols": num_cols,
        "sample_headers": header[:10]
    }


def analyze_metabric():
    print(f"=== ANALIZANDO METABRIC DATASET EN: {METABRIC_DIR} ===\n")
    
    files_to_check = [
        "data_clinical_patient.txt",
        "data_clinical_sample.txt",
        "data_mrna_illumina_microarray.txt",
        "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        "data_cna.txt",
        "data_mutations.txt",
        "data_methylation_promoters_rrbs.txt",
        "data_gene_panel_matrix.txt"
    ]
    
    summary = []
    for filename in files_to_check:
        fp = METABRIC_DIR / filename
        res = inspect_tsv_header_and_shape(fp)
        if res:
            summary.append(res)
            print(f"[FILE] {res['file']} ({res['size_mb']} MB)")
            print(f"   |-- Filas (Genes/Pacientes/Eventos): {res['total_rows']:,}")
            print(f"   |-- Columnas: {res['total_cols']:,}")
            print(f"   +-- Primeras columnas: {res['sample_headers']}\n")

if __name__ == "__main__":
    analyze_metabric()

