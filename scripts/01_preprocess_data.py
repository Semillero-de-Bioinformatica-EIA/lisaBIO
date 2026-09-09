"""
Script 01: Preprocesamiento e Integración Multimodal de Datos.
Wrapper de la fase de preprocesamiento del pipeline unificado.
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=== INICIANDO PREPROCESAMIENTO Y VERIFICACION MULTIMODAL ===")
    # Ejecuta el pipeline validando los manifiestos, lo que también verifica
    # que los datos existan y tengan el formato correcto según el config.
    cmd = [sys.executable, "-m", "dcis_biomarkers.pipeline", "validate-manifest", "--config", "configs/config.yaml"]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n=== PREPROCESAMIENTO COMPLETADO CON ÉXITO ===")
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] El preprocesamiento falló con código {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
