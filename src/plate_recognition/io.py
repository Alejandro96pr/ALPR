"""Entrada/salida local con orientación EXIF y escritura JSON atómica."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from typing import Iterator, TextIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ALPRError
from .types import ImageArray


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ALPRError(f"No existe el archivo: {path}")
    return path.resolve()


def check_outputs(source: Path, *outputs: Path | None) -> None:
    paths = [source.resolve()] + [p.resolve() for p in outputs if p is not None]
    if len(set(paths)) != len(paths):
        raise ALPRError("La entrada y las salidas deben tener rutas distintas.")
    for path in outputs:
        if path is not None and path.exists():
            raise ALPRError(f"La salida ya existe; elige otra ruta: {path}")


def read_image(path: Path) -> ImageArray:
    """Decodifica la primera imagen y aplica EXIF; no redimensiona."""
    require_file(path)
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ALPRError("Usa una imagen de un solo fotograma/página.")
            rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ALPRError(f"Imagen inválida o no compatible: {path}: {exc}") from exc


def write_image(path: Path, image: ImageArray) -> None:
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        raise ALPRError("Formato de imagen de salida no admitido.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, encoded = cv2.imencode(path.suffix, image)
        if not ok:
            raise ALPRError(f"No se pudo codificar {path}")
        encoded.tofile(path)
    except (cv2.error, OSError) as exc:
        raise ALPRError(f"No se pudo guardar {path}: {exc}") from exc


@contextmanager
def atomic_text(path: Path) -> Iterator[TextIO]:
    """Publica el JSON completo únicamente si el proceso termina correctamente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            yield handle
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def write_json(path: Path, data: object) -> None:
    with atomic_text(path) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

