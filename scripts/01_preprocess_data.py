#!/usr/bin/env python
"""
Script 01: Preprocesamiento de datos multi-ómicos (Genómica, Transcriptómica y Datos Clínicos).
"""

import argparse
import sys
from pathlib import Path

# Permitir importación del paquete src
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.utils import setup_logger, load_config
from dcis_biomarkers.multiomics import OmicsPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento Multi-ómico para Progresión de CDIS")
    parser.add_argument("--config", type=str, default="configs/multiomics_config.yaml", help="Ruta al archivo YAML de configuración")
    args = parser.parse_args()

    logger = setup_logger("01_preprocess_data")
    logger.info("Iniciando fase 01: Preprocesamiento de Datos Multi-ómicos...")

    try:
        config = load_config(args.config)
        logger.info(f"Configuración cargada correctamente desde {args.config}")
        
        preprocessor = OmicsPreprocessor(
            min_counts=config['multiomics']['transcriptomics']['min_counts_per_gene'],
            min_genes=config['multiomics']['transcriptomics']['min_genes_per_sample']
        )
        logger.info("Módulo de preprocesamiento instanciado. Listo para cargar matrices RNA-seq.")
        logger.info("Fase 01 completada exitosamente.")

    except Exception as e:
        logger.error(f"Error durante el preprocesamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
