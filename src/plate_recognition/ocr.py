"""OCR Tesseract local con preprocesamiento configurable."""

from pathlib import Path
import re

import cv2

from .config import Config
from .errors import ALPRError
from .types import ImageArray, Reading


def normalize_text(text: str) -> str:
    """Mayúsculas sin espacios/guiones; conserva otros símbolos y alfabetos."""
    return "".join(ch for ch in text.upper() if not ch.isspace() and ch != "-")


class TesseractOCR:
    def __init__(self, config: Config):
        self.config = config
        try:
            import pytesseract
        except ImportError as exc:
            raise ALPRError("Falta OCR: instala pip install -e '.[ocr]' y Tesseract 5.") from exc
        self.engine = pytesseract
        self.engine.pytesseract.tesseract_cmd = config.tesseract_cmd
        self.options = f"--oem 1 --psm {config.psm}"
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", config.language):
            raise ALPRError("Idioma inválido; ejemplo: eng o eng+spa.")
        if config.tessdata_dir:
            directory = Path(config.tessdata_dir).resolve()
            if not directory.is_dir() or '"' in str(directory):
                raise ALPRError(f"Directorio tessdata inválido: {directory}")
            for language in config.language.split("+"):
                if not (directory / f"{language}.traineddata").is_file():
                    raise ALPRError(f"Falta {language}.traineddata en {directory}")
            self.options += f' --tessdata-dir "{directory}"'
        try:
            self.engine.get_tesseract_version()
            available = self.engine.get_languages(config=self.options)
            missing = set(config.language.split("+")) - set(available)
            if missing:
                raise ALPRError(f"Instala los idiomas Tesseract: {', '.join(sorted(missing))}")
        except (OSError, RuntimeError) as exc:
            raise ALPRError(f"No se pudo iniciar Tesseract 5: {exc}") from exc

    def read(self, crop: ImageArray) -> Reading:
        if crop.size == 0:
            raise ALPRError("El recorte OCR está vacío.")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, self.config.ocr_height / gray.shape[0])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if self.config.binarize:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            data = self.engine.image_to_data(gray, lang=self.config.language,
                                             config=self.options,
                                             output_type=self.engine.Output.DICT,
                                             timeout=self.config.ocr_timeout)
        except (OSError, RuntimeError) as exc:
            raise ALPRError(f"Falló la lectura OCR (idioma/pesos/timeout): {exc}") from exc
        tokens = [(text.strip(), float(score))
                  for text, score in zip(data["text"], data["conf"])
                  if text.strip() and float(score) >= 0]
        if not tokens:
            return Reading("", 0.0)
        length = sum(len(text) for text, _ in tokens)
        confidence = sum(len(text) * score for text, score in tokens) / (100 * length)
        return Reading(" ".join(text for text, _ in tokens), max(0.0, min(1.0, confidence)))

