import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

try:
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

def calculate_classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """
    Calcula métricas de clasificación estándar y calibración.
    y_true: (N,) etiquetas binarias o multiclase.
    y_pred_prob: (N, C) o (N,) probabilidades.
    """
    y_pred = np.argmax(y_pred_prob, axis=1) if y_pred_prob.ndim > 1 else (y_pred_prob >= 0.5).astype(int)
    
    if y_pred_prob.ndim == 1 or y_pred_prob.shape[1] == 2:
        prob = y_pred_prob[:, 1] if y_pred_prob.ndim > 1 else y_pred_prob
        try:
            auroc = roc_auc_score(y_true, prob)
            auprc = average_precision_score(y_true, prob)
            brier = brier_score_loss(y_true, prob)
        except ValueError:
            # Solo una clase presente
            auroc, auprc, brier = np.nan, np.nan, np.nan
    else:
        # Multiclase (One-vs-Rest)
        try:
            auroc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
            # AUPRC para multiclase no es directo en scikit-learn con un solo score, 
            # se necesita iterar por clases o usar un macro promedio.
            auprc = np.nan 
            brier = np.nan
        except ValueError:
            auroc = np.nan
            auprc = np.nan
            brier = np.nan

    f1_macro = f1_score(y_true, y_pred, average="macro")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1_macro": f1_macro,
        "brier_score": brier
    }

def calculate_survival_metrics(event_times: np.ndarray, event_observed: np.ndarray, hazard_risk: np.ndarray) -> dict:
    """
    Calcula el C-index de Concordancia para supervivencia.
    No devuelve valores inventados si falla.
    """
    if not HAS_LIFELINES:
        raise ImportError("lifelines is not installed.")
        
    try:
        # hazard_risk es el riesgo (a mayor riesgo, menor tiempo de supervivencia)
        # c-index espera predicciones donde mayores valores significan menores tiempos o mayores tiempos
        # Depende de la convención de la librería. Usualmente, partial hazard = -risk
        c_index = concordance_index(event_times, -hazard_risk, event_observed)
        return {"c_index": c_index}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Fallo calculando C-index: {e}")
        return {"c_index": np.nan}
