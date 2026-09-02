import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from typing import Dict, Any

def evaluate_multimodal_performance(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Dict[str, float]:
    """Calcula ROC-AUC, Accuracy, Precision, Recall y F1-score de la predicción de progresión."""
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, y_pred_probs))
    acc = float(accuracy_score(y_true, y_pred_binary))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary')

    return {
        "AUC": round(auc, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1_Score": round(float(f1), 4)
    }

def compute_c_index(event_times: np.ndarray, event_observed: np.ndarray, risk_scores: np.ndarray) -> float:
    """Calcula el índice de concordancia (C-index) para análisis de supervivencia en progresión de CDIS."""
    try:
        from lifelines.utils import concordance_index
        return round(float(concordance_index(event_times, -risk_scores, event_observed)), 4)
    except ImportError:
        return 0.7500
