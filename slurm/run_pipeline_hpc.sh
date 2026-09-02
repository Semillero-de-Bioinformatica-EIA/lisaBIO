#!/bin/bash
# Script Maestro para enviar la tubería completa a Slurm con dependencias secuenciales

echo "Enviando Pipeline Multimodal de Biomarcadores CDIS a HPC Slurm..."

# 1. Enviar preprocesamiento
JOB1=$(sbatch --parsable slurm/01_preprocess.sh)
echo "Job 1 (Preprocesamiento) enviado: $JOB1"

# 2. Enviar extracción de imágenes (WSI) dependiente de Job 1
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/02_extract_features_gpu.sh)
echo "Job 2 (WSI Features GPU) enviado: $JOB2 (Esperando a $JOB1)"

# 3. Enviar entrenamiento multimodal dependiente de Job 2
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/03_train_multimodal_gpu.sh)
echo "Job 3 (Entrenamiento Multimodal GPU) enviado: $JOB3 (Esperando a $JOB2)"

# 4. Enviar extracción de biomarcadores XAI dependiente de Job 3
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/04_identify_biomarkers.sh)
echo "Job 4 (Biomarcadores XAI) enviado: $JOB4 (Esperando a $JOB3)"

echo "--------------------------------------------------------"
echo "Todos los trabajos han sido encolados exitosamente."
echo "Puedes monitorear el estado con: squeue -u $USER"
