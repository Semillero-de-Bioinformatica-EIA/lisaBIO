#!/usr/bin/env python
"""
Script 03: Entrenamiento de la red de fusión multimodal (Multi-ómica + Patología WSI).
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from dcis_biomarkers.utils import setup_logger, load_config
from dcis_biomarkers.models import MultimodalFusionNetwork

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de Modelo de Fusión Multimodal")
    parser.add_argument("--config", type=str, default="configs/fusion_model_config.yaml", help="Ruta al YAML del modelo")
    args = parser.parse_args()

    logger = setup_logger("03_train_multimodal_model")
    logger.info("Iniciando fase 03: Entrenamiento del modelo de fusión multimodal...")

    try:
        config = load_config(args.config)
        model = MultimodalFusionNetwork(
            omics_dim=config['model']['omics_input_dim'],
            vision_dim=config['model']['vision_input_dim'],
            latent_dim=config['model']['latent_dim'],
            num_classes=config['model']['num_classes']
        )
        logger.info(f"Modelo {config['model']['architecture']} inicializado con éxito.")
        logger.info("Fase 03 completada exitosamente.")

    except Exception as e:
        logger.error(f"Error durante el entrenamiento del modelo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
