import json
from pathlib import Path
from unittest.mock import Mock

import jsonschema
import pytest
from PIL import Image

from plate_recognition import cli
from plate_recognition.config import load_config
from plate_recognition.errors import ALPRError
from plate_recognition.io import atomic_text, read_image
from plate_recognition.pipeline import Pipeline
from plate_recognition.types import Detection

ROOT = Path(__file__).resolve().parents[1]


def validate_result(data):
    jsonschema.validate(data, json.loads((ROOT / "schemas/inference.schema.json").read_text()))


@pytest.mark.parametrize("command", [None, "image", "video", "train", "eval-detector",
                                     "eval-reading", "prepare-data", "synthetic"])
def test_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(([command] if command else []) + ["--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_cli_image_json_schema_and_overlay(tmp_path, monkeypatch, ocr):
    pipeline = Pipeline(Mock(detect=Mock(return_value=[Detection((10, 20, 80, 60), 0.9)])), ocr)
    monkeypatch.setattr(cli, "build_pipeline", lambda config: pipeline)
    output, annotated = tmp_path / "result.json", tmp_path / "annotated.png"
    source = ROOT / "data/synthetic/test_0.png"
    assert cli.main(["image", str(source), "--output", str(output), "--annotated", str(annotated)]) == 0
    result = json.loads(output.read_text())
    validate_result(result)
    assert result["frames"][0]["plates"][0]["text"] == "TEST0"
    assert read_image(annotated).shape == read_image(source).shape


def test_missing_and_invalid_input_before_model_loading(tmp_path, monkeypatch, capsys):
    factory = Mock()
    monkeypatch.setattr(cli, "build_pipeline", factory)
    missing = tmp_path / "absent.png"
    assert cli.main(["image", str(missing), "--output", str(tmp_path / "o.json")]) == 2
    missing.write_text("not an image")
    assert cli.main(["image", str(missing), "--output", str(tmp_path / "o.json")]) == 2
    factory.assert_not_called()
    assert "Error:" in capsys.readouterr().err


def test_exif_orientation(tmp_path):
    image = Image.new("RGB", (40, 20))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    assert read_image(path).shape == (40, 20, 3)


def test_atomic_json_failure_leaves_no_output(tmp_path):
    path = tmp_path / "result.json"
    with pytest.raises(RuntimeError):
        with atomic_text(path) as handle:
            handle.write("partial")
            raise RuntimeError("failed")
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("body", ["unknown: 1", "confidence: .nan", "batch: 0", "normalize: nope",
                                  "[]", "- item", "device: 2", "weights: null"])
def test_invalid_configuration(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(ALPRError):
        load_config(path)


def test_config_path_and_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("weights: model.pt\n")
    assert load_config(path).weights == str(tmp_path / "model.pt")
    assert load_config(path, weights="other.pt").weights == "other.pt"

