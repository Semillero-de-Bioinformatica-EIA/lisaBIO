from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path

@dataclass
class ReproducibilityConfig:
    seed: int = 42
    deterministic_cudnn: bool = True
    num_workers: int = 4

@dataclass
class OutcomeConfig:
    schema: str = "binary" # binary or ternary
    time_window_months: float = 60.0

@dataclass
class DataConfig:
    case_manifest_path: str = "data/manifests/cases.parquet"
    feature_manifest_path: str = "data/manifests/feature_manifest.parquet"
    cache_dir: str = "data/cache"

@dataclass
class ModelConfig:
    omics_embed_dim: int = 256
    vision_embed_dim: int = 256
    fused_dim: int = 512
    dropout: float = 0.3
    vision_backbone: str = "resnet50" # resnet50, conch, uni

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"ce": 1.0, "cox": 0.2})

@dataclass
class PipelineConfig:
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "data/results"

    @classmethod
    def load_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Manually unpack or use dacite/marshmallow for robustness in production.
        # For simplicity here:
        return cls(
            reproducibility=ReproducibilityConfig(**data.get("reproducibility", {})),
            outcome=OutcomeConfig(**data.get("outcome", {})),
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            output_dir=data.get("output_dir", "data/results")
        )
