#!/bin/bash
#SBATCH --job-name=dcis_preprocess
#SBATCH --output=logs/slurm_preprocess_%j.out
#SBATCH --error=logs/slurm_preprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

# --- Carga de Módulos en HPC (Ajustar según el cluster: Slurm / LMOD) ---
module purge
module load python/3.10
# module load gcc/11.2.0

# --- Activar Entorno Conda / Virtualenv ---
source activate dcis-biomarkers || conda activate dcis-biomarkers

echo "=========================================================="
echo "Inicio de Preprocesamiento Multi-ómico en HPC"
echo "Nodo ejecutor: $(hostname)"
echo "Fecha y hora: $(date)"
echo "=========================================================="

mkdir -p logs

# Ejecución del Script 01 CLI (Sin necesidad de Jupyter)
python scripts/01_preprocess_data.py --config configs/multiomics_config.yaml

echo "Preprocesamiento completado exitosamente."
