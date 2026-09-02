#!/bin/bash
#SBATCH --job-name=dcis_wsi_features
#SBATCH --output=logs/slurm_wsi_%j.out
#SBATCH --error=logs/slurm_wsi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# --- Carga de Módulos GPU en HPC ---
module purge
module load python/3.10
module load cuda/11.8
module load openslide/3.4.1

source activate dcis-biomarkers || conda activate dcis-biomarkers

echo "=========================================================="
echo "Extracción de Características WSI en Nodo GPU HPC"
echo "GPU asignada: $CUDA_VISIBLE_DEVICES"
echo "Nodo: $(hostname)"
echo "=========================================================="

mkdir -p logs

# Ejecución del Script 02 CLI
python scripts/02_extract_image_features.py --config configs/pathology_config.yaml

echo "Extracción de parches y embeddings WSI finalizada."
