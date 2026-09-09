import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FeatureExtractor:
    """
    Extracción de características usando un backbone pre-entrenado real.
    Reemplaza la implementación dummy original que devolvía torch.randn.
    """
    def __init__(self, backbone_name: str = "resnet50", device: str = "cpu"):
        self.device = device
        self.backbone_name = backbone_name
        self.model = self._load_backbone()
        self.model.to(self.device)
        self.model.eval()

    def _load_backbone(self) -> nn.Module:
        if self.backbone_name == "resnet50":
            # Usar ResNet50 pre-entrenado en ImageNet
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Eliminar la capa de clasificación (FC)
            modules = list(model.children())[:-1]
            return nn.Sequential(*modules)
        elif self.backbone_name == "conch":
            raise NotImplementedError("CONCH backbone requiere acceso a weights específicos y no está empaquetado por defecto.")
        else:
            raise ValueError(f"Backbone no soportado: {self.backbone_name}")

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) normalizado según el backbone.
        Retorna: (B, Feature_Dim)
        """
        x = x.to(self.device)
        features = self.model(x)
        features = features.view(features.size(0), -1) # Flatten (B, D, 1, 1) -> (B, D)
        return features
