version v3



# DCIS Progression Biomarkers Platform

Plataforma de investigación reproducible para la identificación de biomarcadores moleculares y morfológicos asociados con la progresión de Carcinoma Ductal In Situ (CDIS) a carcinoma invasivo.

## Advertencia Científica
Esta es una plataforma de **investigación** en desarrollo. Ningún modelo, métrica o biomarcador generado por esta herramienta ha sido validado clínicamente. No utilizar para la toma de decisiones clínicas.

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

## Uso

El pipeline completo se gestiona a través de una CLI central.

### 1. Pruebas Técnicas
Puedes verificar la instalación con datos sintéticos:
```bash
pytest tests/ -v
python -m dcis_biomarkers.pipeline validate-manifest --synthetic
```

### 2. Ejecución Real
La configuración del pipeline se define en `configs/config.yaml`.

```bash
# Validar contrato de datos
python -m dcis_biomarkers.pipeline validate-manifest --config configs/config.yaml

# Entrenar el modelo
python -m dcis_biomarkers.pipeline train --config configs/config.yaml

# Generar reporte de biomarcadores
python -m dcis_biomarkers.pipeline export-biomarkers --config configs/config.yaml
```

## Estructura
- `src/dcis_biomarkers/`: Paquete principal.
  - `data/`: Contratos de datos, manifiestos y partición.
  - `models/`: Encoders, MIL y redes de fusión.
  - `multiomics/`: Adaptadores modulares por ómica.
  - `pathology/`: Procesamiento WSI, segmentación y tiling.
  - `xai/`: Explicabilidad de parches y SHAP tabular.
  - `evaluation/`: Métricas de clasificación, supervivencia y calibración.
