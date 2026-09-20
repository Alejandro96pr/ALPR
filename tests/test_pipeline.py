from unittest.mock import Mock

import numpy as np
import pytest

from plate_recognition.errors import ALPRError
from plate_recognition.pipeline import Pipeline, annotate
from plate_recognition.types import Detection


def test_empty_detection_does_not_call_ocr(frame, ocr):
    pipeline = Pipeline(Mock(detect=Mock(return_value=[])), ocr)
    assert pipeline.run(frame) == []
    ocr.read.assert_not_called()


def test_clips_boxes_and_preserves_dimensions(frame, ocr):
    detector = Mock(detect=Mock(return_value=[Detection((-4, 10, 200, 70), 0.5)]))
    result = Pipeline(detector, ocr).run(frame)
    assert result[0].bbox == (0, 10, 160, 70)
    assert result[0].text == "TEST0"
    assert result[0].raw_text == "TEST-0"
    assert result[0].confidence == pytest.approx(0.4)
    assert ocr.read.call_args.args[0].shape == (60, 160, 3)
    assert annotate(frame, result).shape == frame.shape
    assert np.all(frame == 127)


@pytest.mark.parametrize("box", [(10, 10, 1, 1), (200, 0, 220, 10), (0, 0, float("nan"), 2)])
def test_invalid_boxes_are_ignored(frame, ocr, box):
    assert Pipeline(Mock(detect=Mock(return_value=[Detection(box, 0.5)])), ocr).run(frame) == []
    ocr.read.assert_not_called()


def test_invalid_confidence_and_input(frame, ocr):
    pipeline = Pipeline(Mock(detect=Mock(return_value=[Detection((0, 0, 20, 20), float("nan"))])), ocr)
    with pytest.raises(ALPRError, match="confianza"):
        pipeline.run(frame)
    with pytest.raises(ALPRError, match="BGR"):
        pipeline.run(np.zeros((0, 0, 3), dtype=np.uint8))

