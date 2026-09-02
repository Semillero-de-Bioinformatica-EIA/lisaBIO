import json
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Carga un archivo de configuración YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo de configuración {config_path} no existe.")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_json(data: Dict[str, Any], filepath: str, indent: int = 4) -> None:
    """Guarda un diccionario como un archivo JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(filepath: str) -> Dict[str, Any]:
    """Carga un archivo JSON."""
    path = Path(filepath)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
