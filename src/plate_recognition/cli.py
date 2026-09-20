"""Interfaz de línea de comandos; ayuda disponible sin dependencias de modelos."""

import argparse
from pathlib import Path
import sys

from .config import load_config
from .errors import ALPRError


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="ALPR local: YOLO + OCR Tesseract")
    sub = root.add_subparsers(dest="command", required=True)
    for name, help_text in (("image", "Inferir una imagen"), ("video", "Procesar todos los frames"),
                            ("train", "Ajustar detector con datos propios"),
                            ("eval-detector", "Precision, recall y mAP del detector"),
                            ("eval-reading", "Exactitud y CER de OCR y extremo a extremo")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, help="Archivo YAML de configuración")
        command.add_argument("--weights", help="Checkpoint YOLO .pt local")
        command.add_argument("--device", help="cpu, mps o índice CUDA, p. ej. 0")
        if name in ("image", "video", "eval-reading"):
            command.add_argument("--language", help="Idioma Tesseract, p. ej. eng o eng+spa")
            command.add_argument("--confidence", type=float, help="Umbral de detección")
            command.add_argument("--output", type=Path, required=True, help="JSON de salida (nuevo)")
        if name in ("image", "video"):
            command.add_argument("input", type=Path)
            command.add_argument("--annotated", type=Path, help="Imagen/vídeo anotado opcional")
        if name in ("train", "eval-detector"):
            command.add_argument("--data", type=Path, required=True, help="dataset.yaml local")
            command.add_argument("--project", help="Carpeta de experimentos")
            command.add_argument("--name", dest="run_name", help="Nombre nuevo del experimento")
            command.add_argument("--batch", type=int)
            command.add_argument("--image-size", type=int)
        if name == "train":
            command.add_argument("--epochs", type=int)
            command.add_argument("--seed", type=int)
            command.add_argument("--learning-rate", type=float)
        if name == "eval-detector":
            command.add_argument("--split", choices=("train", "val", "test"), default="test")
        if name == "eval-reading":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--mode", choices=("ocr", "e2e"), default="ocr")
            command.add_argument("--iou", type=float, default=0.5)
    prepare = sub.add_parser("prepare-data", help="Convertir manifiesto y dividir por grupos")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-fraction", type=float, default=0.7)
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    synthetic = sub.add_parser("synthetic", help="Generar fixtures artificiales de pruebas")
    synthetic.add_argument("--destination", type=Path, required=True)
    return root


def build_pipeline(config):
    from .detector import YOLODetector
    from .ocr import TesseractOCR
    from .pipeline import Pipeline

    return Pipeline(YOLODetector(config), TesseractOCR(config), config.normalize)


def execute(args: argparse.Namespace) -> None:
    if args.command == "synthetic":
        from .synthetic import generate_fixtures

        print(generate_fixtures(args.destination))
        return
    if args.command == "prepare-data":
        from .data import prepare_dataset

        print(prepare_dataset(args.manifest, args.destination, args.seed,
                              args.train_fraction, args.val_fraction))
        return
    keys = ("weights", "device", "language", "confidence", "project", "run_name", "batch",
            "image_size", "epochs", "seed", "learning_rate")
    config = load_config(args.config, **{key: getattr(args, key, None) for key in keys})
    if args.command in ("train", "eval-detector"):
        from .training import run_detector

        print(run_detector(config, args.data, evaluate=args.command == "eval-detector",
                           split=getattr(args, "split", "test")))
        return
    from .io import check_outputs, read_image, require_file, write_image, write_json

    if args.command == "eval-reading":
        from .data import load_manifest
        from .evaluation import evaluate_reading
        from .ocr import TesseractOCR

        check_outputs(args.manifest, args.output)
        load_manifest(args.manifest)  # Valida entradas antes de inicializar motores.
        pipeline = build_pipeline(config) if args.mode == "e2e" else None
        ocr = pipeline.ocr if pipeline else TesseractOCR(config)
        result = evaluate_reading(args.manifest, ocr, pipeline, args.iou, config.normalize)
        write_json(args.output, result)
    else:
        require_file(args.input)
        check_outputs(args.input, args.output, args.annotated)
        if args.command == "image":
            from .pipeline import annotate, frame_result

            image = read_image(args.input)
            pipeline = build_pipeline(config)
            plates = pipeline.run(image)
            if args.annotated:
                write_image(args.annotated, annotate(image, plates))
            write_json(args.output, {"schema_version": "1.0", "media_type": "image",
                                    "source": str(args.input), "frames": [frame_result(image, plates)]})
        else:
            from .video import process_video

            process_video(args.input, args.output, build_pipeline(config), args.annotated)
    print(args.output)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        execute(args)
        return 0
    except (ALPRError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.", file=sys.stderr)
        return 130

