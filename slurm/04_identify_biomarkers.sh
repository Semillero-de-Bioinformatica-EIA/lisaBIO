#!/bin/bash
#SBATCH --job-name=dcis_xai_biomarkers
#SBATCH --output=logs/slurm_xai_%j.out
#SBATCH --error=logs/slurm_xai_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

module purge
module load python/3.10

source activate dcis-biomarkers || conda activate dcis-biomarkers

echo "=========================================================="
echo "Extracción de Biomarcadores XAI en HPC"
echo "=========================================================="

mkdir -p logs

python scripts/04_identify_biomarkers.py --output data/results/biomarkers/

echo "Biomarcadores generados."
