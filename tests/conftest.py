"""La suite básica bloquea conexiones y no importa modelos reales."""

import socket
from unittest.mock import Mock

import numpy as np
import pytest

from plate_recognition.types import Reading


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Las pruebas no pueden acceder a la red")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


@pytest.fixture
def frame():
    return np.full((80, 160, 3), 127, dtype=np.uint8)


@pytest.fixture
def ocr():
    return Mock(read=Mock(return_value=Reading("TEST-0", 0.8)))

