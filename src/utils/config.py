import json
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image
import sys, os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

try:
    import yaml
except Exception:
    yaml = None

# --------------------------------------------------------------------------------------
# Raíz del proyecto (asume este archivo en <repo>/src/utils/config.py)
# --------------------------------------------------------------------------------------
def _find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

ROOT_DIR: Path = _find_project_root()
CONFIG_DIR: Path = ROOT_DIR / "config"


def _read_yaml_config(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _api_config() -> Dict[str, Any]:
    path = Path(os.environ.get("API_CONFIG_PATH", CONFIG_DIR / "api.yaml"))
    if not path.is_absolute():
        path = ROOT_DIR / path
    return _read_yaml_config(path)


_API_CONFIG = _api_config()
_API_SECTION = _API_CONFIG.get("api", {}) if isinstance(_API_CONFIG.get("api"), dict) else {}
_PATHS_SECTION = _API_CONFIG.get("paths", {}) if isinstance(_API_CONFIG.get("paths"), dict) else {}


def _cfg_path(key: str, default: str) -> Path:
    raw = os.environ.get(key.upper(), _PATHS_SECTION.get(key, default))
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT_DIR / path


def _cfg_tuple(env_key: str, yaml_key: str, default: list[str]) -> Tuple[str, ...]:
    if os.environ.get(env_key):
        return tuple(x.strip().lower() for x in os.environ[env_key].split(",") if x.strip())
    values = _API_SECTION.get(yaml_key, default)
    return tuple(str(x).strip().lower() for x in values)


def _cfg_value(env_key: str, yaml_key: str, default: Any, caster):
    if os.environ.get(env_key) is not None:
        return caster(os.environ[env_key])
    return caster(_API_SECTION.get(yaml_key, default))

# --------------------------------------------------------------------------------------
# Configuración única (NO duplicar clases ni get_config() más abajo)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    ROOT: Path = ROOT_DIR
    CONFIG_DIR: Path = CONFIG_DIR

    # Paths de datos/modelo (resueltos relativo al repo)
    PET_STATS_PATH: Path = _cfg_path("pet_stats", "data/pet_stats.json")
    MODEL_META_PATH: Path = _cfg_path("model_meta", "models/meta.json")
    MODEL_WEIGHTS_PATH: Path = _cfg_path("model_weights", "resnet101/model_trained/ResNet101.pth")
    MODEL_YAML_PATH: Path = _cfg_path("model_config", "config/model/oxford_pets_binary_resnet101.yaml")
    RESNET_SRC_DIR: Path = _cfg_path("resnet_src", "resnet101/src")

    # Parámetros de API / IO 
    ALLOWED_EXTS: Tuple[str, ...] = _cfg_tuple("ALLOWED_EXTS", "allowed_exts", [".jpg", ".jpeg", ".png"])
    ALLOWED_MIMES: Tuple[str, ...] = _cfg_tuple("ALLOWED_MIMES", "allowed_mimes", ["image/jpeg", "image/png"])
    MAX_IMAGE_MB: int = _cfg_value("MAX_IMAGE_MB", "max_image_mb", 5, int)
    TIMEOUT_CONNECT: float = _cfg_value("TIMEOUT_CONNECT", "timeout_connect", 5.0, float)
    TIMEOUT_READ: float = _cfg_value("TIMEOUT_READ", "timeout_read", 10.0, float)

    # Preprocesamiento / device policy 
    RESIZE_SCALE: float = _cfg_value("RESIZE_SCALE", "resize_scale", 1.14, float)
    DEVICE_POLICY: str = str(os.environ.get("DEVICE_POLICY", _API_SECTION.get("device_policy", "auto")))  # 'auto' | 'cpu' | 'cuda'

# Singleton en memoria (una sola instancia compartida)
_CFG: Optional[Config] = None

def get_config() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config()
    return _CFG


# --------------------------------------------------------------------------------------
# Utilidades de stats (acepta pet_stats.json o meta.json con normalization.mean/std)
# --------------------------------------------------------------------------------------
def load_pet_stats(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Carga estadísticas de normalización desde:
      - data/pet_stats.json con llaves {loc, scale, img_size?}, o
      - models/meta.json con {"normalization": {"mean","std"}, "input_size": ...}
    Devuelve dict normalizado: {"loc": [...], "scale": [...], "img_size": 224}
    """
    cfg = get_config()
    stats_path = Path(path) if path is not None else cfg.PET_STATS_PATH
    if not stats_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de stats en: {stats_path}")
    raw = json.loads(stats_path.read_text(encoding="utf-8"))

    # Caso A: esquema pet_stats.json
    if "loc" in raw and "scale" in raw:
        if "img_size" not in raw:
            raw["img_size"] = 224
        return {"loc": raw["loc"], "scale": raw["scale"], "img_size": int(raw["img_size"])}

    # Caso B: esquema meta.json con normalization
    norm = raw.get("normalization", {})
    mean, std = norm.get("mean"), norm.get("std")
    if isinstance(mean, list) and isinstance(std, list):
        return {
            "loc": mean,
            "scale": std,
            "img_size": int(raw.get("input_size", 224)),}

    raise ValueError("El archivo de stats no contiene 'loc/scale' ni 'normalization.mean/std'.")


def mostrar_imagen(ruta):
    """
    Muestra la imagen ubicada en `ruta`.
    Acepta formatos comunes (jpg, png, jpeg, etc.).
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No encontré el archivo: {ruta}")
    try:
        img = Image.open(ruta).convert("RGB")
    except OSError as e:
        raise ValueError(f"El archivo no parece ser una imagen válida:\n{e}")

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(os.path.basename(ruta))
    plt.axis("off")
    plt.show()
