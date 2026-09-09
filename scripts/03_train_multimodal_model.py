"""
Script 03: Entrenamiento y Fusión Multimodal (WSI + Omics).
Wrapper de la fase de entrenamiento del pipeline unificado.
"""

import sys
import subprocess

def main():
    print("=== ENTRENAMIENTO DE MODELO DE FUSIÓN MULTIMODAL ===")
    cmd = [sys.executable, "-m", "dcis_biomarkers.pipeline", "train", "--config", "configs/config.yaml"]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n=== ENTRENAMIENTO COMPLETADO ===")
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] El entrenamiento falló con código {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
