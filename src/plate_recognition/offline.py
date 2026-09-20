"""Impide conexiones Python durante llamadas a Ultralytics (CLI monohilo)."""

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch


def _denied(*args: object, **kwargs: object) -> None:
    raise OSError("Red deshabilitada: proporciona todos los recursos en rutas locales.")


@contextmanager
def offline() -> Iterator[None]:
    """Bloquea sockets y DNS; no sustituye a un firewall para código no confiable."""
    with patch("socket.socket.connect", _denied), patch("socket.socket.connect_ex", _denied), \
            patch("socket.create_connection", _denied), patch("socket.getaddrinfo", _denied):
        yield

