import json
from unittest.mock import Mock

import pytest

from plate_recognition.data import load_manifest, prepare_dataset, validate_dataset
from plate_recognition.errors import ALPRError
from plate_recognition.evaluation import edit_distance, evaluate_reading, reading_metrics
from plate_recognition.pipeline import Pipeline
from plate_recognition.synthetic import generate_fixtures
from plate_recognition.types import Detection, Reading


def test_group_split_and_yolo_conversion(tmp_path):
    manifest = generate_fixtures(tmp_path / "source")
    first = prepare_dataset(manifest, tmp_path / "one")
    second = prepare_dataset(manifest, tmp_path / "two")
    assert validate_dataset(first) == first.parent
    assert json.loads((first.parent / "split.json").read_text()) == json.loads(
        (second.parent / "split.json").read_text())
    groups = []
    for split in ("train", "val", "test"):
        records = load_manifest(first.parent / f"{split}.json")
        groups.append({record["group"] for record in records})
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    label = next(first.parent.glob("labels/*/000000.txt")).read_text().split()
    assert list(map(float, label)) == pytest.approx([0, 0.5, 85 / 160, 200 / 320, 50 / 160])


def test_invalid_annotations_and_duplicate(tmp_path):
    manifest = generate_fixtures(tmp_path / "source")
    entries = json.loads(manifest.read_text())
    entries[0]["plates"][0]["bbox"] = [10, 0, 500, 50]
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ALPRError, match="Caja"):
        load_manifest(manifest)
    entries[0]["plates"] = []
    entries.append(entries[0])
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ALPRError, match="duplicada"):
        load_manifest(manifest)


def test_distance_cer_and_empty_denominators():
    assert edit_distance("TEST", "TSET") == 2
    assert reading_metrics([("ABC", "AXC"), ("D", "")])["cer"] == 0.5
    assert reading_metrics([])["exact_accuracy"] is None
    assert reading_metrics([("", "X")])["cer"] is None
    assert reading_metrics([("A", "ABCDE")])["cer"] == 4


def test_evaluation_penalizes_misses_and_duplicate_predictions(tmp_path):
    manifest = generate_fixtures(tmp_path / "source")
    entries = json.loads(manifest.read_text())[:1]
    entries[0]["plates"].append({"bbox": [0, 0, 40, 20], "text": "MISSED"})
    manifest.write_text(json.dumps(entries))
    ocr = Mock(read=Mock(return_value=Reading("TEST-0", 1.0)))
    detector = Mock(detect=Mock(return_value=[Detection((60, 60, 260, 110), 0.9),
                                             Detection((60, 60, 260, 110), 0.8)]))
    result = evaluate_reading(manifest, ocr, Pipeline(detector, ocr))
    det = result["detection_at_iou"]
    assert (det["true_positives"], det["false_positives"], det["false_negatives"]) == (1, 1, 1)
    assert result["ocr_matched_detections"]["exact_accuracy"] == 1
    assert result["end_to_end"]["exact_plate_recall"] == 0.5
    assert result["end_to_end"]["exact_plate_precision"] == 0.5

