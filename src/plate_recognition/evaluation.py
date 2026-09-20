"""Métricas de lectura en cajas reales y extremo a extremo con emparejamiento IoU."""

from pathlib import Path

from .data import load_manifest
from .io import read_image
from .ocr import normalize_text
from .pipeline import Pipeline, clipped_box
from .types import OCR


def edit_distance(reference: str, hypothesis: str) -> int:
    """Distancia de Levenshtein con memoria O(longitud de hipótesis)."""
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def reading_metrics(pairs: list[tuple[str, str]]) -> dict:
    total = len(pairs)
    chars = sum(len(reference) for reference, _ in pairs)
    errors = sum(edit_distance(reference, hypothesis) for reference, hypothesis in pairs)
    return {"samples": total, "exact_accuracy": sum(a == b for a, b in pairs) / total if total else None,
            "cer": errors / chars if chars else None, "character_errors": errors,
            "reference_characters": chars}


def iou(a: tuple | list, b: tuple | list) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_reading(manifest: Path, ocr: OCR, pipeline: Pipeline | None = None,
                     threshold: float = 0.5, normalize: bool = True) -> dict:
    """OCR sobre GT; opcionalmente lectura end-to-end, con omisiones y falsos positivos."""
    if not 0 < threshold <= 1:
        from .errors import ALPRError

        raise ALPRError("El umbral IoU debe estar en (0, 1].")
    records = load_manifest(manifest)
    transform = normalize_text if normalize else lambda text: text
    crop_pairs, matched_pairs, all_pairs = [], [], []
    tp = fp = fn = exact = 0
    for record in records:
        image = read_image(Path(record["image"]))
        truth = record["plates"]
        for plate in truth:
            x1, y1, x2, y2 = clipped_box(plate["bbox"], image.shape[1], image.shape[0])
            reading = ocr.read(image[y1:y2, x1:x2])
            crop_pairs.append((transform(plate["text"]), transform(reading.text)))
        if pipeline is None:
            continue
        predictions = sorted(pipeline.run(image), key=lambda plate: plate.detection_confidence,
                             reverse=True)
        remaining = set(range(len(truth)))
        for prediction in predictions:
            match = max(remaining, key=lambda i: iou(prediction.bbox, truth[i]["bbox"]), default=None)
            if match is None or iou(prediction.bbox, truth[match]["bbox"]) < threshold:
                fp += 1
                continue
            remaining.remove(match)
            tp += 1
            pair = (transform(truth[match]["text"]), transform(prediction.raw_text))
            matched_pairs.append(pair)
            all_pairs.append(pair)
            exact += pair[0] == pair[1]
        fn += len(remaining)
        all_pairs.extend((transform(truth[i]["text"]), "") for i in sorted(remaining))
    result = {"schema_version": "1.0", "images": len(records), "normalized": normalize,
              "ocr_ground_truth_crops": reading_metrics(crop_pairs)}
    if pipeline is not None:
        result.update({"detection_at_iou": {"iou": threshold, "true_positives": tp,
                       "false_positives": fp, "false_negatives": fn,
                       "precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / (tp + fn) if tp + fn else None},
                       "ocr_matched_detections": reading_metrics(matched_pairs),
                       "end_to_end": {**reading_metrics(all_pairs),
                                      "exact_plate_recall": exact / (tp + fn) if tp + fn else None,
                                      "exact_plate_precision": exact / (tp + fp) if tp + fp else None}})
    return result

