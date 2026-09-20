"""Fixtures geométricos artificiales; no son datos para medir precisión real."""

from pathlib import Path

import cv2
import numpy as np

from .errors import ALPRError
from .io import write_image, write_json


def generate_fixtures(destination: Path) -> Path:
    """Crea tres carteles TEST y un negativo, sin matrículas fotografiadas."""
    if destination.exists():
        raise ALPRError(f"El destino ya existe: {destination}")
    destination.mkdir(parents=True)
    records = []
    for i in range(4):
        image = np.full((160, 320, 3), 55 + i * 10, dtype=np.uint8)
        plates = []
        if i < 3:
            x, y, w, h = 60 + i * 5, 60, 200, 50
            cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), -1)
            text = f"TEST-{i}"
            cv2.putText(image, text, (x + 12, y + 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 2, cv2.LINE_AA)
            plates = [{"bbox": [x, y, x + w, y + h], "text": text}]
        filename = f"test_{i}.png"
        write_image(destination / filename, image)
        records.append({"image": filename, "group": f"synthetic-{i % 3}", "plates": plates})
    write_json(destination / "manifest.json", records)
    return destination / "manifest.json"
