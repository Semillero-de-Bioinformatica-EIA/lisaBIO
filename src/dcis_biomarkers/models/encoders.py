import torch
import torch.nn as nn

class GenericOmicsEncoder(nn.Module):
    """
    Encoder genérico para cualquier modalidad ómica 1D (Transcriptómica, Mutaciones, etc).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si batch=1, BatchNorm1d puede fallar, manejamos eso
        if x.size(0) == 1 and self.training:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)

class ClinicalEncoder(nn.Module):
    """
    Encoder específico para variables clínicas categóricas (one-hot) y numéricas.
    """
    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
