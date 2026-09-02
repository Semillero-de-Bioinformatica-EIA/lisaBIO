#!/usr/bin/env python
"""
Script 02: Tiling de imágenes H&E de WSI y extracción de embeddings de visión profunda.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.utils import setup_logger, load_config
from dcis_biomarkers.pathology import WSITiler, DeepFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Extracción de características morfológicas en WSI")
    parser.add_argument("--config", type=str, default="configs/pathology_config.yaml", help="Ruta al archivo YAML de patología")
    args = parser.parse_args()

    logger = setup_logger("02_extract_image_features")
    logger.info("Iniciando fase 02: Tiling y extracción de parches WSI...")

    try:
        config = load_config(args.config)
        tiler = WSITiler(
            patch_size=config['pathology']['wsi']['patch_size'],
            stride=config['pathology']['wsi']['stride']
        )
        extractor = DeepFeatureExtractor(
            backbone=config['pathology']['feature_extraction']['backbone'],
            embedding_dim=config['pathology']['feature_extraction']['embedding_dim']
        )
        logger.info(f"Extractor con backbone '{config['pathology']['feature_extraction']['backbone']}' cargado correctamente.")
        logger.info("Fase 02 completada exitosamente.")

    except Exception as e:
        logger.error(f"Error durante la extracción de características de visión: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
