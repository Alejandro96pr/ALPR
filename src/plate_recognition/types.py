"""Contratos de los componentes y resultados serializables."""

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class Reading:
    text: str
    confidence: float


@dataclass(frozen=True)
class Plate:
    bbox: tuple[int, int, int, int]
    text: str
    raw_text: str
    detection_confidence: float
    ocr_confidence: float
    confidence: float

    def to_dict(self) -> dict:
        """xyxy en píxeles, esquina inferior exclusiva."""
        result = asdict(self)
        result["bbox"] = list(self.bbox)
        return result


class Detector(Protocol):
    def detect(self, image: ImageArray) -> list[Detection]:
        """Devuelve cajas sobre la imagen original BGR."""
        ...


class OCR(Protocol):
    def read(self, crop: ImageArray) -> Reading:
        """Lee un recorte BGR y devuelve confianza en [0, 1]."""
        ...

