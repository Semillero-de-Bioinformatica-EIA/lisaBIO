# Biomarcadores de Progresión de CDIS a Carcinoma Invasivo de Mama (AI & Multi-ómica)

Este proyecto proporciona una plataforma integral basada en **Inteligencia Artificial Multimodal** y **Análisis Multi-ómico** para identificar biomarcadores moleculares y morfológicos asociados con la transición del Carcinoma Ductal In Situ (CDIS) a Carcinoma Ductal Invasivo (CDI) de mama.

---

## 🔬 Arquitectura del Proyecto

El sistema integra dos fuentes principales de datos biomédicos:

1. **Datos Moleculares / Multi-ónicos**:
   - Transcriptómica (bulk RNA-seq y transcriptómica espacial).
   - Genómica (mutaciones somáticas, CNVs).
   - Análisis de expresión diferencial (DESeq2, Scanpy), enriquecimiento de rutas metabólicas (GSEA/Reactome) y supervivencia (Kaplan-Meier, Cox Proportional Hazards).

2. **Patología Digital / Morfología (Imágenes WSI)**:
   - Procesamiento de imágenes de lámina completa (*Whole Slide Images* - WSI).
   - Segmentación de tejido y parches morfológicos (*tiling*).
   - Extracción de características profundas mediante *Foundation Models* de visión médica (ej. CONCH, UNI, PLIP, ResNet).
   - Grafos de microambiente espacial (interacción tumor-estroma-TILs).

3. **Fusión Multimodal e IA Explicable (XAI)**:
   - Modelos de Deep Learning con mecanismos de atención multimodal.
   - Ponderación e identificación de biomarcadores significativos mediante **SHAP** y **Mapas de Atención Inter-modal**.

---

## 📁 Estructura de Directorios

```text
dcis-progression-biomarkers/
├── configs/            # Archivos YAML de configuración (parámetros de omica, patología e IA)
├── data/               # Estructura de datos (raw, interim, processed, external, results)
├── notebooks/          # Cuadernos Jupyter analíticos (EDA, entrenamiento, XAI)
├── scripts/            # Scripts CLI ejecutables del pipeline
├── src/dcis_biomarkers # Paquete de código fuente en Python
│   ├── utils/          # Utilidades (logging, IO)
│   ├── multiomics/     # Procesamiento genómico y transcriptómico
│   ├── pathology/      # Procesamiento de WSI y visión por computadora
│   ├── models/         # Encoders y arquitectura de fusión multimodal
│   ├── xai/            # Inteligencia Artificial Explicable (SHAP, mapas de atención)
│   └── evaluation/     # Métricas y evaluación de supervivencia
└── tests/              # Pruebas unitarias
```

---

## 🛠️ Instalación y Configuración

### Entorno Conda
```bash
conda env create -f environment.yml
conda activate dcis-biomarkers
```

### Instalación de Paquete Local en Modo Desarrollo
```bash
pip install -e .
```

---

## 🚀 Flujo de Ejecución del Pipeline

1. **Preprocesamiento Multi-ómico y Limpieza**:
   ```bash
   python scripts/01_preprocess_data.py --config configs/multiomics_config.yaml
   ```

2. **Segmentación y Extracción de Características de WSI**:
   ```bash
   python scripts/02_extract_image_features.py --config configs/pathology_config.yaml
   ```

3. **Entrenamiento de Modelo de Fusión Multimodal**:
   ```bash
   python scripts/03_train_multimodal_model.py --config configs/fusion_model_config.yaml
   ```

4. **Identificación de Biomarcadores y Generación de Reporte XAI**:
   ```bash
   python scripts/04_identify_biomarkers.py --output data/results/biomarkers/
   ```

---

## 📜 Licencia y Citación
Proyecto de Investigación Biomédica. Todos los derechos reservados.
