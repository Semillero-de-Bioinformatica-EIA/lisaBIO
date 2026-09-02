#!/bin/bash
#SBATCH --job-name=dcis_train_multimodal
#SBATCH --output=logs/slurm_train_%j.out
#SBATCH --error=logs/slurm_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

module purge
module load python/3.10
module load cuda/11.8

source activate dcis-biomarkers || conda activate dcis-biomarkers

echo "=========================================================="
echo "Entrenamiento de Modelo Multimodal en GPU HPC"
echo "=========================================================="

mkdir -p logs

python scripts/03_train_multimodal_model.py --config configs/fusion_model_config.yaml

echo "Entrenamiento finalizado."
