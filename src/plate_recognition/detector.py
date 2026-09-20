"""Adaptador YOLO con pesos locales y carga opcional."""

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


class YOLODetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = load_yolo(config.weights)
        if config.class_id not in self.model.names:
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
