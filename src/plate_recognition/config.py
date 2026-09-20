"""Configuración estricta: las rutas YAML son relativas al propio archivo."""

from dataclasses import dataclass, fields
from pathlib import Path
import math
from typing import Any

import yaml

from .errors import ALPRError


@dataclass(frozen=True)
class Config:
    weights: str = "models/plate.pt"
    device: str = "cpu"
    confidence: float = 0.25
    nms_iou: float = 0.45
    image_size: int = 640
    class_id: int = 0
    language: str = "eng"
    tessdata_dir: str | None = None
    tesseract_cmd: str = "tesseract"
    psm: int = 7
    ocr_timeout: float = 10.0
    ocr_height: int = 64
    binarize: bool = False
    normalize: bool = True
    seed: int = 42
    epochs: int = 50
    batch: int = 8
    learning_rate: float = 0.001
    project: str = "outputs/training"
    run_name: str = "plates"

    def validate(self) -> "Config":
        """Rechaza valores mal tipados o fuera de rango antes de cargar modelos."""
        for name in ("confidence", "nms_iou", "learning_rate", "ocr_timeout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ALPRError(f"{name} debe ser un número finito positivo.")
        if self.confidence > 1 or self.nms_iou > 1 or self.learning_rate > 1:
            raise ALPRError("confidence, nms_iou y learning_rate deben estar en (0, 1].")
        for name in ("image_size", "ocr_height", "epochs", "batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ALPRError(f"{name} debe ser un entero positivo.")
        for name in ("class_id", "seed", "psm"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ALPRError(f"{name} debe ser un entero no negativo.")
        if self.psm not in (6, 7, 8, 11, 13):
            raise ALPRError("psm admitidos: 6, 7, 8, 11, 13.")
        for name in ("weights", "device", "language", "tesseract_cmd", "project", "run_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ALPRError(f"{name} debe ser texto no vacío.")
        if self.tessdata_dir is not None and not isinstance(self.tessdata_dir, str):
            raise ALPRError("tessdata_dir debe ser una ruta o null.")
        for name in ("binarize", "normalize"):
            if type(getattr(self, name)) is not bool:
                raise ALPRError(f"{name} debe ser true o false.")
        if Path(self.run_name).name != self.run_name or self.run_name in (".", ".."):
            raise ALPRError("run_name debe ser un nombre simple de carpeta.")
        return self


def load_config(path: Path | None = None, **overrides: Any) -> Config:
    """Carga YAML y aplica overrides CLI (relativos al directorio de ejecución)."""
    values: dict[str, Any] = {}
    if path is not None:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            values = {} if loaded is None else loaded
        except (OSError, yaml.YAMLError) as exc:
            raise ALPRError(f"No se puede leer la configuración {path}: {exc}") from exc
        if not isinstance(values, dict):
            raise ALPRError("La configuración debe ser un objeto YAML.")
        for key in ("weights", "tessdata_dir", "project"):
            if isinstance(values.get(key), str):
                values[key] = str((path.resolve().parent / values[key]).resolve())
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ALPRError(f"Claves de configuración desconocidas: {sorted(unknown, key=str)}")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**values).validate()
