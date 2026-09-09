from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ClassDefinition:
    label_value: int
    name: str
    description: str

@dataclass
class OutcomeDefinition:
    """
    Definición explícita del desenlace clínico.
    """
    name: str
    classes: Dict[int, ClassDefinition]
    description: str
    time_window_months: Optional[float] = None
    censorship_rules: Optional[str] = None
    survival_available: bool = False

    def validate_labels(self, labels: list[int]) -> bool:
        valid_labels = set(self.classes.keys())
        return all(l in valid_labels for l in labels)


BINARY_SCHEMA = OutcomeDefinition(
    name="DCIS_Binary_Progression",
    description="Esquema binario: CDIS indolente vs progresivo",
    classes={
        0: ClassDefinition(0, "DCIS_Indolent", "CDIS no progresivo o indolente"),
        1: ClassDefinition(1, "DCIS_Progressive", "CDIS con progresión a carcinoma invasivo")
    },
    survival_available=True
)

TERNARY_SCHEMA = OutcomeDefinition(
    name="DCIS_Ternary_Progression",
    description="Esquema ternario: CDIS indolente, CDIS progresivo, Carcinoma Invasivo",
    classes={
        0: ClassDefinition(0, "DCIS_Indolent", "CDIS indolente"),
        1: ClassDefinition(1, "DCIS_Progressive", "CDIS progresivo (DCISM)"),
        2: ClassDefinition(2, "Invasive_Carcinoma", "Carcinoma Invasivo (IDC)")
    },
    survival_available=True
)
