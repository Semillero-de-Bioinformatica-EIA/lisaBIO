"""
Script 04: Identificación de Biomarcadores y Mapas de Atención (Pat-XAI).
Wrapper de la fase de XAI del pipeline unificado.
"""

import sys
import subprocess

def main():
    print("=== EXTRACCIÓN DE BIOMARCADORES Y EXPLICABILIDAD XAI ===")
    cmd1 = [sys.executable, "-m", "dcis_biomarkers.pipeline", "export-biomarkers", "--config", "configs/config.yaml"]
    cmd2 = [sys.executable, "-m", "dcis_biomarkers.pipeline", "export-heatmaps", "--config", "configs/config.yaml"]
    
    try:
        subprocess.run(cmd1, check=True)
        subprocess.run(cmd2, check=True)
        print("\n=== EXTRACCIÓN XAI COMPLETADA ===")
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] La extracción XAI falló con código {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
