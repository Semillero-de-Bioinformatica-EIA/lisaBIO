version v3



# DCIS Progression Biomarkers Platform

Plataforma de investigación reproducible para la identificación de biomarcadores moleculares y morfológicos asociados con la progresión de Carcinoma Ductal In Situ (CDIS) a carcinoma invasivo.


## Características
- Integración de Whole Slide Images (WSI) y datos Multi-ómicos (Transcriptómica, CNA, Mutaciones, Clínica).
- Manejo explícito del contrato de datos mediante manifiestos Parquet.
- Prevención estricta de fuga de datos mediante validación cruzada estratificada a nivel de paciente.
- Arquitectura basada en Gated Attention MIL (Multiple Instance Learning).
- Reportes XAI (Explicabilidad) sin fallback a valores aleatorios/sintéticos.

## Instalación
Requiere Python 3.10+.
```bash
# Clonar el repositorio
git clone <url-repo>
cd dcis-progression-biomarkers

# Instalar dependencias base
pip install -e .

# Para desarrollo y pruebas
pip install -e ".[dev]"
```

| Verificación         | Resultado                             |
| -------------------- | ------------------------------------- |
| Compilación Python   | Correcta                              |
| Instalación editable | Correcta                              |
| Pruebas              | Fallan                                |
| Black                | Falla: 240 archivos requieren formato |
| Ruff                 | La ejecución de CI no queda limpia    |
| CI de GitHub         | Fallaría por formato y pruebas        |



## Área
Estado actual
Evaluación
Organización del repositorio
Mejorada
Buena separación de módulos
Configuración
Parcialmente correcta
Falta validación completa de campos
Manifiestos
Implementados
Requieren más validaciones y conexión al pipeline
Prevención de fuga
Implementada parcialmente
La división por paciente es correcta, pero debe integrarse al entrenamiento
Multi-ómica
Arquitectura inicial
Adaptadores presentes, pipeline de entrenamiento ausente
TIFF/WSI
Parcial
Reader inicial, tiling y metadatos aún incompletos
H&E
Parcial
Segmentación presente, Macenko pendiente
MIL
Defectuoso
Falla la normalización de atención
Fusión multimodal
Defectuosa
Error de dimensiones
XAI
Mejorada
Ya no fabrica valores, pero SHAP es demasiado genérico
CLI
Incompleto
La mayoría de comandos terminan en pass
Tests
Insuficientes
Hay pruebas nuevas, pero dos fallan
CI
Configurado
Actualmente fallaría por formato y tests
Reproducibilidad
Parcial
Semillas y config presentes, entrenamiento no conectado
