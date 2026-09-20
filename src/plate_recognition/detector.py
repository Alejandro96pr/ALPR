"""Adaptador YOLO con pesos locales y carga opcional."""

import hashlib
import os
from pathlib import Path
import sys
from typing import Any

from .config import Config
from .errors import ALPRError
from .io import require_file
from .offline import offline
from .types import Detection, ImageArray


def load_yolo(weights: str) -> Any:
    path = require_file(Path(weights))
    if path.suffix.lower() != ".pt":
        raise ALPRError("El adaptador requiere un checkpoint YOLO .pt local y confiable.")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    cache = Path(sys.prefix) / ".alpr-cache"
    cache.mkdir(parents=True, exist_ok=True)
    for variable, directory in (("YOLO_CONFIG_DIR", "ultralytics"),
                                ("MPLCONFIGDIR", "matplotlib"), ("TORCH_HOME", "torch"),
                                ("XDG_CACHE_HOME", "xdg")):
        location = Path(os.environ.setdefault(variable, str(cache / directory)))
        location.mkdir(parents=True, exist_ok=True)
    try:
        with offline():
            from ultralytics import YOLO, settings

            settings.update({"sync": False, "hub": False, "wandb": False,
                             "clearml": False, "comet": False, "mlflow": False,
                             "neptune": False, "raytune": False, "tensorboard": False,
                             "dvc": False})
            return YOLO(str(path), task="detect")
    except ImportError as exc:
        raise ALPRError("Falta el detector: instala pip install -e '.[detector]'.") from exc
    except Exception as exc:
        raise ALPRError(f"No se pudieron cargar los pesos YOLO locales: {exc}") from exc


def model_classes(model: Any) -> dict[int, str]:
    """Normaliza los nombres de clase expuestos por distintas versiones de YOLO."""
    names = getattr(model, "names", None)
    if isinstance(names, (list, tuple)):
        classes = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            classes = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise ALPRError("El modelo contiene índices de clase inválidos.") from exc
    else:
        raise ALPRError("El modelo no declara sus clases de detección.")
    if not classes or any(index < 0 or not name.strip() for index, name in classes.items()):
        raise ALPRError("El modelo no contiene clases de detección válidas.")
    return dict(sorted(classes.items()))


def inspect_yolo_model(weights: str, class_id: int) -> dict:
    """Carga un checkpoint local y devuelve información suficiente para integrarlo."""
    path = Path(weights).resolve()
    model = load_yolo(str(path))
    task = getattr(model, "task", "detect")
    if task != "detect":
        raise ALPRError(f"El modelo usa la tarea '{task}', pero se necesita detección.")
    classes = model_classes(model)
    if class_id not in classes:
        raise ALPRError(f"class_id={class_id} no existe; disponibles: {list(classes)}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": "1.0",
        "compatible": True,
        "weights": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "task": task,
        "classes": [{"id": index, "name": name} for index, name in classes.items()],
        "selected_class": {"id": class_id, "name": classes[class_id]},
    }


class YOLODetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = load_yolo(config.weights)
        self.classes = model_classes(self.model)
        if config.class_id not in self.classes:
            raise ALPRError(f"class_id={config.class_id} no existe en el modelo.")

    def detect(self, image: ImageArray) -> list[Detection]:
        """Ultralytics devuelve xyxy ya escalado al tamaño original."""
        try:
            with offline():
                result = self.model.predict(source=image, conf=self.config.confidence,
                                            iou=self.config.nms_iou,
                                            imgsz=self.config.image_size,
                                            device=self.config.device,
                                            classes=[self.config.class_id],
                                            verbose=False, save=False)[0]
            if result.boxes is None:
                return []
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            return [Detection(tuple(box), float(score)) for box, score in zip(boxes, scores)]
        except Exception as exc:
            raise ALPRError(f"Falló la detección YOLO: {exc}") from exc
