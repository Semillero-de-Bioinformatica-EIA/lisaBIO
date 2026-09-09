import numpy as np
import logging

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

def perform_shap_analysis(model, background_data, target_data):
    """
    Realiza análisis SHAP para modelos tabulares o modalidades ómicas.
    A diferencia de la versión anterior, NUNCA devuelve datos inventados.
    """
    if not HAS_SHAP:
        logger.error("SHAP no está instalado.")
        return {"status": "unavailable", "shap_values": None, "error": "SHAP not installed"}

    try:
        # Dependiendo del tipo de modelo, se elige el explainer
        # Si es deep learning: DeepExplainer o GradientExplainer
        # Si es tabular (RF/XGBoost): TreeExplainer
        
        # Como es genérico, intentaremos usar KernelExplainer o DeepExplainer
        # Por simplicidad, KernelExplainer (lento pero agnóstico)
        explainer = shap.KernelExplainer(model, background_data)
        shap_values = explainer.shap_values(target_data)
        
        return {
            "status": "success",
            "shap_values": shap_values,
            "expected_value": explainer.expected_value
        }
    except Exception as e:
        logger.error(f"Fallo en análisis SHAP: {e}")
        # Retornamos error explícito, NUNCA valores aleatorios
        return {"status": "failed", "shap_values": None, "error": str(e)}
