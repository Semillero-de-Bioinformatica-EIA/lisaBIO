"""
Script 02: Extracción de Características WSI con Self-Supervised Models.
Wrapper de la fase de extracción de features del pipeline unificado.
"""

import sys
import subprocess

def main():
    print("=== EXTRACCIÓN DE CARACTERÍSTICAS WSI (Self-Supervised / CONCH) ===")
    cmd = [sys.executable, "-m", "dcis_biomarkers.pipeline", "extract-features", "--config", "configs/config.yaml"]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n=== EXTRACCIÓN DE CARACTERÍSTICAS COMPLETADA ===")
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] La extracción de características falló con código {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
