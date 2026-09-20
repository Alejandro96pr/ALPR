"""Ajuste/evaluación de detector con parámetros registrados y sin descargas."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform
import random

import numpy as np
import yaml

from .config import Config
from .data import validate_dataset
from .detector import load_yolo
from .errors import ALPRError
from .io import write_json
from .offline import offline


def run_detector(config: Config, dataset: Path, *, evaluate: bool = False,
                 split: str = "test") -> Path:
    """Ejecuta entrenamiento o validación y devuelve la carpeta del experimento."""
    root = validate_dataset(dataset)
    run = (Path(config.project) / config.run_name).resolve()
    if run.exists():
        raise ALPRError(f"El experimento ya existe; cambia run_name/--name: {run}")
    model = load_yolo(config.weights)
    if evaluate and (len(model.names) != 1 or 0 not in model.names or config.class_id != 0):
        raise ALPRError("eval-detector requiere un detector de clase única (índice 0); "
                        "ajusta el modelo con alpr train antes de medir este dataset.")
    run.mkdir(parents=True)
    # Congela rutas absolutas para que Ultralytics no aplique su datasets_dir global.
    local_data = run / "dataset.yaml"
    local_data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train",
                                         "val": "images/val", "test": "images/test",
                                         "names": {0: "plate"}}), encoding="utf-8")
    parameters = {"config": asdict(config), "dataset": str(dataset.resolve()),
                  "mode": "val" if evaluate else "train", "split": split if evaluate else "train",
                  "python": platform.python_version(),
                  "versions": {name: version(name) for name in ("ultralytics", "torch", "numpy")}}
    write_json(run / "parameters.json", parameters)
    random.seed(config.seed)
    np.random.seed(config.seed)
    try:
        with offline():
            common = dict(data=str(local_data), device=config.device, imgsz=config.image_size,
                          batch=config.batch, project=str(run.parent), name=run.name,
                          exist_ok=True, plots=False, workers=0)
            if evaluate:
                result = model.val(**common, split=split)
            else:
                result = model.train(**common, epochs=config.epochs, seed=config.seed,
                                     deterministic=True, lr0=config.learning_rate, optimizer="AdamW",
                                     amp=False, pretrained=True, single_cls=True,
                                     fliplr=0.0, flipud=0.0)
        metrics = result.results_dict if result is not None else {}
        write_json(run / "metrics.json", {"mode": parameters["mode"],
                                         "metrics": {str(k): float(v) for k, v in metrics.items()}})
    except Exception as exc:
        write_json(run / "error.json", {"error": str(exc)})
        raise ALPRError(f"Falló {parameters['mode']}; consulta {run}: {exc}") from exc
    return run
