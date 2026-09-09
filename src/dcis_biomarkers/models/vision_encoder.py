import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    """
    Encoder visual modular que envuelve el FeatureExtractor pre-entrenado.
    """
    def __init__(self, feature_dim: int = 2048, embed_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # Reducción de dimensionalidad
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Bag_Size, Feature_Dim)
        """
        B, N, D = x.shape
        x_flat = x.view(B * N, D)
        proj_flat = self.proj(x_flat)
        return proj_flat.view(B, N, self.embed_dim)
