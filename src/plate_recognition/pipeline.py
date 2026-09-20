"""Localización, recorte y lectura independientes de los proveedores de modelos."""

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .errors import ALPRError
from .ocr import normalize_text
from .types import Detector, ImageArray, OCR, Plate


def clipped_box(box: tuple, width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def score(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ALPRError("El componente devolvió una confianza fuera de [0, 1].")
    return float(value)


class Pipeline:
    def __init__(self, detector: Detector, ocr: OCR, normalize: bool = True):
        self.detector, self.ocr, self.normalize = detector, ocr, normalize

    def run(self, image: ImageArray) -> list[Plate]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ALPRError("Se necesita una imagen BGR uint8 no vacía de tres canales.")
        height, width = image.shape[:2]
        plates = []
        for detection in self.detector.detect(image):
            box = clipped_box(detection.bbox, width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            reading = self.ocr.read(image[y1:y2, x1:x2].copy())
            det, rec = score(detection.confidence), score(reading.confidence)
            text = normalize_text(reading.text) if self.normalize else reading.text
            plates.append(Plate(box, text, reading.text, det, rec, det * rec))
        return plates


def annotate(image: ImageArray, plates: list[Plate]) -> ImageArray:
    """Dibuja sobre una copia sin cambiar dimensiones ni orientación."""
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    for plate in plates:
        x1, y1, x2, y2 = plate.bbox
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline="lime", width=2)
        # La fuente incluida por Pillow cubre el alfabeto latino; JSON conserva Unicode.
        label = f"{plate.text} {plate.confidence:.2f}"
        draw.text((x1, max(0, y1 - 16)), label, fill="lime", stroke_width=1, stroke_fill="black")
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def frame_result(image: ImageArray, plates: list[Plate], frame_index: int | None = None,
                 timestamp_seconds: float | None = None) -> dict:
    return {"frame_index": frame_index, "timestamp_seconds": timestamp_seconds,
            "width": image.shape[1], "height": image.shape[0],
            "plates": [plate.to_dict() for plate in plates]}

