"""Manifiesto común y conversión reproducible a YOLO, agrupada por vehículo/sesión."""

import json
import math
from pathlib import Path
import random
import shutil
import tempfile

import yaml

from .errors import ALPRError
from .io import read_image, require_file, write_image, write_json


def load_manifest(path: Path) -> list[dict]:
    """Valida anotaciones xyxy en coordenadas de la imagen orientada por EXIF."""
    require_file(path)
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ALPRError(f"Manifiesto JSON inválido: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise ALPRError("El manifiesto debe ser una lista no vacía.")
    seen: set[Path] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("image"), str):
            raise ALPRError("Cada registro necesita image, group y plates.")
        image_path = (path.resolve().parent / record["image"]).resolve()
        if image_path in seen:
            raise ALPRError(f"Imagen duplicada en el manifiesto: {image_path}")
        seen.add(image_path)
        if not isinstance(record.get("group"), str) or not record["group"].strip():
            raise ALPRError("Cada registro necesita group (vehículo/sesión) no vacío.")
        if not isinstance(record.get("plates"), list):
            raise ALPRError("plates debe ser una lista, vacía para imágenes negativas.")
        image = read_image(image_path)
        height, width = image.shape[:2]
        for plate in record["plates"]:
            if (not isinstance(plate, dict) or not isinstance(plate.get("text"), str)
                    or not plate["text"].strip()):
                raise ALPRError("Cada matrícula necesita bbox y text no vacío.")
            box = plate.get("bbox")
            if not isinstance(box, list) or len(box) != 4 or not all(
                type(v) in (int, float) and math.isfinite(v) for v in box
            ):
                raise ALPRError("bbox debe contener cuatro números finitos xyxy.")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ALPRError(f"Caja fuera de imagen o degenerada: {image_path}: {box}")
        record["image"] = str(image_path)
    return records


def prepare_dataset(manifest: Path, destination: Path, seed: int = 42,
                    train_fraction: float = 0.7, val_fraction: float = 0.2) -> Path:
    """Divide grupos completos y convierte imágenes a PNG orientado + etiquetas YOLO."""
    if destination.exists():
        raise ALPRError(f"El destino ya existe: {destination}")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
            train_fraction + val_fraction < 1):
        raise ALPRError("Las fracciones train/val deben ser positivas y sumar menos de 1.")
    records = load_manifest(manifest)
    groups = sorted({record["group"] for record in records})
    if len(groups) < 3:
        raise ALPRError("Se necesitan al menos tres grupos para train/val/test sin fuga.")
    random.Random(seed).shuffle(groups)
    n_train = max(1, min(len(groups) - 2, int(len(groups) * train_fraction)))
    n_val = max(1, min(len(groups) - n_train - 1, int(len(groups) * val_fraction)))
    splits = {group: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
              for i, group in enumerate(groups)}
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="alpr-data-", dir=destination.parent))
    try:
        manifests: dict[str, list] = {name: [] for name in ("train", "val", "test")}
        for split in manifests:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        for index, record in enumerate(records):
            split = splits[record["group"]]
            filename = f"{index:06d}.png"
            relative = Path("images") / split / filename
            image = read_image(Path(record["image"]))
            height, width = image.shape[:2]
            write_image(staging / relative, image)
            lines = []
            for plate in record["plates"]:
                x1, y1, x2, y2 = plate["bbox"]
                lines.append(f"0 {(x1+x2)/(2*width):.10f} {(y1+y2)/(2*height):.10f} "
                             f"{(x2-x1)/width:.10f} {(y2-y1)/height:.10f}\n")
            (staging / "labels" / split / f"{index:06d}.txt").write_text("".join(lines))
            manifests[split].append({**record, "image": relative.as_posix()})
        for split, entries in manifests.items():
            write_json(staging / f"{split}.json", entries)
        yaml_data = {"path": str(destination), "train": "images/train", "val": "images/val",
                     "test": "images/test", "names": {0: "plate"}}
        (staging / "dataset.yaml").write_text(yaml.safe_dump(yaml_data), encoding="utf-8")
        write_json(staging / "split.json", {"seed": seed, "groups": splits,
                   "train_fraction": train_fraction, "val_fraction": val_fraction,
                   "counts": {s: len(r) for s, r in manifests.items()}})
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination / "dataset.yaml"


def validate_dataset(path: Path) -> Path:
    """Admite el formato de carpetas producido por prepare-data, sin scripts download."""
    require_file(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ALPRError(f"Dataset YAML inválido: {exc}") from exc
    if not isinstance(data, dict) or "download" in data:
        raise ALPRError("Dataset inválido: no se permiten instrucciones download.")
    if data.get("names") not in ({0: "plate"}, ["plate"]):
        raise ALPRError("El dataset debe definir una sola clase: names: {0: plate}.")
    root_value = data.get("path", ".")
    if not isinstance(root_value, str):
        raise ALPRError("path debe ser una ruta local.")
    root = (path.resolve().parent / root_value).resolve()
    for split in ("train", "val", "test"):
        expected = f"images/{split}"
        if data.get(split) != expected:
            raise ALPRError(f"Se requiere {split}: {expected}; usa prepare-data.")
        images = sorted(p for p in (root / expected).glob("*") if p.is_file())
        if not images:
            raise ALPRError(f"Partición vacía o inexistente: {root / expected}")
        stems: set[str] = set()
        positive_boxes = 0
        for image in images:
            if image.stem in stems:
                raise ALPRError(f"Nombres de imagen ambiguos: {image.stem}")
            stems.add(image.stem)
            read_image(image)
            label = require_file(root / "labels" / split / f"{image.stem}.txt")
            for line in label.read_text().splitlines():
                try:
                    cls, x, y, w, h = map(float, line.split())
                except ValueError as exc:
                    raise ALPRError(f"Etiqueta inválida: {label}") from exc
                if not (all(math.isfinite(v) for v in (cls, x, y, w, h)) and cls == 0 and
                        0 < w <= 1 and 0 < h <= 1 and
                        w / 2 - 1e-8 <= x <= 1 - w / 2 + 1e-8 and
                        h / 2 - 1e-8 <= y <= 1 - h / 2 + 1e-8):
                    raise ALPRError(f"Etiqueta no normalizada o fuera de imagen: {label}")
                positive_boxes += 1
        if not positive_boxes:
            raise ALPRError(f"La partición {split} no contiene matrículas anotadas; "
                            "no permite entrenar/medir mAP. Revisa la división por grupos.")
    return root
