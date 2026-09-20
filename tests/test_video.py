import json
from unittest.mock import Mock

import cv2
import pytest

from plate_recognition.errors import ALPRError
from plate_recognition.pipeline import Pipeline
from plate_recognition.video import process_video
from test_io_cli import validate_result


def test_video_round_trip(tmp_path, frame, ocr):
    source, output, annotated = tmp_path / "in.avi", tmp_path / "out.json", tmp_path / "out.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 80))
    assert writer.isOpened(), "OpenCV necesita soporte MJPG para esta prueba"
    for _ in range(3):
        writer.write(frame)
    writer.release()
    pipeline = Pipeline(Mock(detect=Mock(return_value=[])), ocr)
    assert process_video(source, output, pipeline, annotated) == 3
    result = json.loads(output.read_text())
    validate_result(result)
    assert [f["timestamp_seconds"] for f in result["frames"]] == [0.0, 0.1, 0.2]
    assert all(f["plates"] == [] for f in result["frames"])
    capture = cv2.VideoCapture(str(annotated))
    try:
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        ok, decoded = capture.read()
        assert ok and decoded.shape == frame.shape
    finally:
        capture.release()


def test_invalid_video_and_output_collision(tmp_path, ocr):
    source = tmp_path / "bad.mp4"
    source.write_text("invalid")
    pipeline = Pipeline(Mock(), ocr)
    with pytest.raises(ALPRError, match="abrir"):
        process_video(source, tmp_path / "out.json", pipeline)
    assert not (tmp_path / "out.json").exists()
    with pytest.raises(ALPRError, match="distintas"):
        process_video(source, source, pipeline)


def test_existing_partial_video_is_not_deleted(tmp_path, frame, ocr):
    source = tmp_path / "in.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 80))
    assert writer.isOpened()
    writer.write(frame)
    writer.release()
    partial = tmp_path / "out.partial.avi"
    partial.write_bytes(b"preserve this existing file")
    pipeline = Pipeline(Mock(detect=Mock(return_value=[])), ocr)
    with pytest.raises(ALPRError, match="temporal ya existe"):
        process_video(source, tmp_path / "out.json", pipeline, tmp_path / "out.avi")
    assert partial.read_bytes() == b"preserve this existing file"
    assert not (tmp_path / "out.json").exists()
