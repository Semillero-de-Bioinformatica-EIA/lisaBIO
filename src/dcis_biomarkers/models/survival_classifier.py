import torch
import torch.nn as nn

class SurvivalHazardPredictor(nn.Module):
    """Predicción de función de riesgo (hazard ratio) para análisis de tiempo a progresión invasiva (Cox Loss)."""

    def __init__(self, input_dim: int = 128):
        super(SurvivalHazardPredictor, self).__init__()
        self.hazard_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False)
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        # Retorna el log-hazard ratio
        return self.hazard_head(fused_features)
