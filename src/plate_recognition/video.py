"""Procesamiento secuencial de vídeo con JSON incremental y memoria acotada."""

import json
import math
from pathlib import Path

import cv2

from .errors import ALPRError
from .io import atomic_text, check_outputs, require_file
from .pipeline import Pipeline, annotate, frame_result


def process_video(source: Path, output: Path, pipeline: Pipeline,
                  annotated: Path | None = None) -> int:
    """Procesa todos los frames. Vídeo anotado sin audio, a FPS constante de origen."""
    require_file(source)
    check_outputs(source, output, annotated)
    if annotated is not None and annotated.suffix.lower() not in (".avi", ".mp4"):
        raise ALPRError("El vídeo de salida debe terminar en .avi o .mp4.")
    capture = cv2.VideoCapture(str(source))
    writer = None
    temp_video = None
    count = 0
    try:
        if not capture.isOpened():
            raise ALPRError(f"No se puede abrir el vídeo: {source}")
        rotation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
        auto_rotation = capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        if rotation and not auto_rotation:
            raise ALPRError("El backend no puede aplicar la orientación del vídeo.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ALPRError("FPS del vídeo desconocidos o inválidos.")
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = None
        with atomic_text(output) as handle:
            header = {"schema_version": "1.0", "media_type": "video", "source": str(source),
                      "fps": fps, "timestamp_basis": "frame_index/fps"}
            handle.write(json.dumps(header, ensure_ascii=False)[:-1] + ', "frames": [')
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if size is None:
                    size = (width, height)
                    if annotated is not None:
                        if width % 2 or height % 2:
                            raise ALPRError("Vídeo de dimensiones impares: exporta solo JSON; "
                                            "OpenCV puede truncar un píxel al codificar.")
                        annotated.parent.mkdir(parents=True, exist_ok=True)
                        candidate = annotated.with_name(annotated.stem + ".partial" + annotated.suffix)
                        if candidate.exists():
                            raise ALPRError(f"El archivo temporal ya existe: {candidate}")
                        temp_video = candidate
                        codec = "FFV1" if annotated.suffix.lower() == ".avi" else "mp4v"
                        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*codec), fps, size)
                        if not writer.isOpened():
                            raise ALPRError(f"No se puede crear vídeo con códec {codec}.")
                elif size != (width, height):
                    raise ALPRError("El vídeo cambia de dimensiones entre fotogramas.")
                plates = pipeline.run(frame)
                if count:
                    handle.write(",")
                json.dump(frame_result(frame, plates, count, count / fps), handle,
                          ensure_ascii=False, allow_nan=False)
                if writer is not None:
                    writer.write(annotate(frame, plates))
                count += 1
            if count == 0:
                raise ALPRError("El vídeo no contiene fotogramas decodificables.")
            if expected > 0 and count < expected:
                raise ALPRError(f"Vídeo incompleto: decodificados {count} de {expected} frames.")
            handle.write(f'], "frame_count": {count}}}\n')
            if writer is not None:
                writer.release()
                writer = None
                # Confirma que el contenedor se puede decodificar y conserva tamaño/número.
                verify = cv2.VideoCapture(str(temp_video))
                try:
                    verified = 0
                    while True:
                        ok, decoded = verify.read()
                        if not ok:
                            break
                        if (decoded.shape[1], decoded.shape[0]) != size:
                            raise ALPRError("El codificador cambió las dimensiones de salida.")
                        verified += 1
                    if verified != count:
                        raise ALPRError("El vídeo de salida no contiene todos los fotogramas.")
                finally:
                    verify.release()
                temp_video.replace(annotated)
        return count
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if temp_video is not None:
            temp_video.unlink(missing_ok=True)
