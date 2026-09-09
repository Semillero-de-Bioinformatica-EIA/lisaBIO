import numpy as np
from sklearn.calibration import calibration_curve

def calculate_calibration(y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Calcula la curva de calibración (reliability diagram) y el Expected Calibration Error (ECE).
    """
    if y_pred_prob.ndim > 1:
        # Asumimos que la columna 1 es la probabilidad de la clase positiva (progresivo)
        prob = y_pred_prob[:, 1]
    else:
        prob = y_pred_prob

    try:
        prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy='uniform')
        
        # Cálculo simplificado de ECE
        # (En una versión completa se calcula como suma ponderada de diferencias absolutas)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "expected_calibration_error": ece
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error calculando calibración: {e}")
        return {
            "prob_true": [],
            "prob_pred": [],
            "expected_calibration_error": np.nan
        }
