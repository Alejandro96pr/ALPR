import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plate_recognition.config import Config
from plate_recognition.data import prepare_dataset
from plate_recognition.detector import YOLODetector, load_yolo
from plate_recognition.errors import ALPRError
from plate_recognition.ocr import TesseractOCR
from plate_recognition.synthetic import generate_fixtures
from plate_recognition.training import run_detector


def test_missing_weights_before_import(tmp_path):
    with pytest.raises(ALPRError, match="No existe"):
        load_yolo(str(tmp_path / "missing.pt"))


def test_optional_dependency_messages(tmp_path, monkeypatch):
    path = tmp_path / "local.pt"
    path.write_bytes(b"test-double")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ALPRError, match="detector"):
        load_yolo(str(path))
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    with pytest.raises(ALPRError, match="OCR"):
        TesseractOCR(Config())


def test_yolo_adapter_uses_local_array_and_scales_no_boxes(tmp_path, monkeypatch, frame):
    path = tmp_path / "local.pt"
    path.write_bytes(b"test-double")
    boxes = SimpleNamespace(xyxy=Mock(), conf=Mock())
    boxes.xyxy.cpu.return_value.tolist.return_value = [[1, 2, 80, 40]]
    boxes.conf.cpu.return_value.tolist.return_value = [0.75]
    model = Mock(names={0: "plate"})
    model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    settings = Mock()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Mock(return_value=model), settings=settings))
    detector = YOLODetector(Config(weights=str(path)))
    result = detector.detect(frame)
    assert result[0].bbox == (1, 2, 80, 40)
    assert result[0].confidence == 0.75
    assert model.predict.call_args.kwargs["source"] is frame
    assert settings.update.call_args.args[0]["sync"] is False


def test_ocr_adapter_and_confidence(monkeypatch, frame):
    engine = Mock()
    engine.get_languages.return_value = ["eng"]
    engine.image_to_data.return_value = {"text": ["", "TEST", "0"], "conf": [-1, 90, 50]}
    monkeypatch.setitem(sys.modules, "pytesseract", engine)
    result = TesseractOCR(Config()).read(frame)
    assert result.text == "TEST 0"
    assert result.confidence == pytest.approx(0.82)
    engine.get_languages.return_value = []
    with pytest.raises(ALPRError, match="idiomas"):
        TesseractOCR(Config())


@pytest.mark.parametrize("evaluate", [False, True])
def test_train_validation_dispatch_and_logs(tmp_path, monkeypatch, evaluate):
    from plate_recognition import training

    data = prepare_dataset(generate_fixtures(tmp_path / "source"), tmp_path / "dataset")
    model = Mock(names={0: "plate"})
    metrics = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.0})
    model.train.return_value = metrics
    model.val.return_value = metrics
    monkeypatch.setattr(training, "load_yolo", lambda weights: model)
    monkeypatch.setattr(training, "version", lambda name: "test-double")
    config = Config(project=str(tmp_path / "runs"), epochs=1)
    result = run_detector(config, data, evaluate=evaluate)
    assert (result / "parameters.json").is_file()
    assert (result / "metrics.json").is_file()
    if evaluate:
        assert model.val.call_args.kwargs["split"] == "test"
    else:
        assert model.train.call_args.kwargs["amp"] is False
        assert model.train.call_args.kwargs["deterministic"] is True
        assert model.train.call_args.kwargs["seed"] == 42
    with pytest.raises(ALPRError, match="existe"):
        run_detector(config, data)
