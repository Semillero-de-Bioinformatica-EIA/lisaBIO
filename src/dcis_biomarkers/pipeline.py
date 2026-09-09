import argparse
import sys
import os
import torch
import numpy as np
import random
from pathlib import Path

from dcis_biomarkers.config import PipelineConfig
from dcis_biomarkers.data.manifest import validate_manifest
from dcis_biomarkers.data.synthetic import generate_synthetic_paired_dataset

def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="DCIS Progression Biomarkers Pipeline")
    parser.add_argument("command", choices=["train", "evaluate", "extract-features", "export-biomarkers", "export-heatmaps", "validate-manifest"], help="Command to run")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--synthetic", action="store_true", help="Generate and use synthetic data for testing")
    
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    config = PipelineConfig.load_yaml(config_path)
    set_seed(config.reproducibility.seed, config.reproducibility.deterministic_cudnn)

    # Synthetic data generation for technical tests
    if args.synthetic:
        print(">> Generando datos sintéticos para pruebas técnicas...")
        synth_dir = Path("data/synthetic_test")
        generate_synthetic_paired_dataset(synth_dir)
        config.data.case_manifest_path = str(synth_dir / "cases.parquet")
        config.data.feature_manifest_path = str(synth_dir / "feature_manifest.parquet")

    if args.command == "validate-manifest":
        print(f">> Validando manifiestos...")
        report = validate_manifest(
            config.data.case_manifest_path, 
            config.data.feature_manifest_path, 
            Path(config.output_dir) / "manifest_validation_report.json"
        )
        print(">> Resultado de validación:", report["status"])
        if report["errors"]:
            print(">> Errores:", report["errors"])
            sys.exit(1)

    elif args.command == "train":
        print(">> Iniciando pipeline de entrenamiento...")
        # Lógica de entrenamiento usando los módulos refactorizados
        pass

    elif args.command == "evaluate":
        print(">> Evaluando modelo...")
        pass

    elif args.command == "extract-features":
        print(">> Extrayendo características de imágenes...")
        pass

    elif args.command == "export-biomarkers":
        print(">> Exportando biomarcadores XAI...")
        pass

    elif args.command == "export-heatmaps":
        print(">> Generando heatmaps de atención ROI...")
        pass

if __name__ == "__main__":
    main()
