# ALPR local con Python, YOLO y Tesseract

Proyecto modular para localizar matrículas, recortarlas y leerlas en imágenes y
vídeos locales. Devuelve cajas, texto y confianza en JSON, con anotaciones
visuales opcionales. **No incluye un detector entrenado para matrículas ni afirma
ninguna precisión real.** Aporta pesos con licencia adecuada o ajusta YOLO con tus
datos antes de evaluar el sistema en el entorno de destino.

## Arquitectura y decisiones

`imagen BGR → Detector → cajas xyxy → recortes → OCR → resultados → JSON/anotación`

- **Ultralytics YOLO11 / PyTorch**, con checkpoint `.pt` local. Se recomienda
  ajustar YOLO11n preentrenado en COCO para reducir recursos; COCO no tiene la
  clase matrícula. Un checkpoint ALPR existente y autorizado permite inferir
  directamente. `class_id` selecciona la clase de matrícula de ese checkpoint.
- **Tesseract 5 mediante pytesseract**, alternativa a PaddleOCR: OCR local LSTM
  sobre recortes, con idiomas y archivos `traineddata` explícitos. Se usa PSM 7
  para una línea; PSM 6 puede servir para dos líneas. No se instala Paddle ni un
  segundo runtime de aprendizaje profundo. Esta elección prioriza instalación
  sencilla y control de recursos; debe compararse con otros OCR en datos propios.
- **OpenCV** para preprocesamiento, vídeo y codificación; **Pillow** para aplicar
  orientación EXIF antes de calcular cajas y para dibujar texto. No se cambia el
  tamaño visual de las imágenes. En un JPEG con EXIF rotado, ancho y alto reflejan
  la imagen orientada, no el orden del raster almacenado.
- Contratos `Detector` y `OCR` en `types.py`, inyectados en `Pipeline`. Para cambiar
  proveedor implementa `detect(image)` o `read(crop)` y cambia la factoría
  `build_pipeline` de la CLI. Las pruebas usan estos contratos sin modelos.
- Configuración YAML estricta y sobrescrituras CLI. Las rutas YAML son relativas
  al YAML; las rutas CLI son relativas al directorio donde se ejecuta el comando.

La documentación oficial se consultó el 20-09-2026. Se fijan versiones concretas
compatibles, sin exigir la última versión: Python **3.11–3.13**, NumPy 2.2.6,
OpenCV 4.12.0.88, Pillow 12.0.0, PyYAML 6.0.3, Ultralytics 8.3.221, PyTorch 2.9.0,
torchvision 0.24.0 y pytesseract 0.3.13. No mezcles `opencv-python` con
`opencv-python-headless`/`opencv-contrib-python`: comparten `cv2`. Se elige
`opencv-python` porque Ultralytics ya lo requiere.

## Archivos

```text
configs/default.yaml             # Inferencia y valores predeterminados
configs/train.yaml               # Ajuste desde pesos COCO locales
data/README.md
data/synthetic/                  # 4 PNG y manifest.json artificiales
models/README.md                 # Origen y requisitos de pesos externos
outputs/                        # Resultados locales, excluidos de Git
schemas/inference.schema.json    # Contrato JSON versión 1.0
src/plate_recognition/
  __init__.py, __main__.py
  cli.py, config.py, errors.py, types.py
  detector.py, ocr.py, pipeline.py
  io.py, offline.py, video.py
  data.py, synthetic.py, training.py, evaluation.py
tests/                          # Pruebas unitarias y de integración sin modelos
pyproject.toml
.gitignore
README.md
```

## Instalación: siempre en entorno virtual

Desde la raíz del proyecto, con Python 3.11 instalado:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
alpr --help
```

En PowerShell activa con `.venv\Scripts\Activate.ps1`. Para la inferencia y el
entrenamiento reales añade los extras dentro del mismo entorno:

```bash
python -m pip install -e '.[detector,ocr]'
python -m pip check
```

`dev` instala pytest, jsonschema y ruff. La instalación base no instala torch,
Ultralytics ni pytesseract. **Instalar paquetes sí necesita red**, salvo que
dispongas de un repositorio de wheels local. Ni instalación ni pruebas descargan
pesos. Para registrar todas las versiones transitivas de tu máquina:

```bash
python -m pip freeze > outputs/environment.txt
```

### Ejecutable OCR y datos de idioma

`pytesseract` es un adaptador: necesitas además **Tesseract 5** y al menos el modelo
LSTM `eng.traineddata`. No se instalan silenciosamente. Para mantener también el
ejecutable nativo aislado puedes usar un entorno Conda dedicado, **en lugar de**
`.venv` (no hace falta crear ambos):

```bash
conda create -p .venv -c conda-forge python=3.11 pip tesseract=5
conda activate ./.venv
python -m pip install -e '.[dev,detector,ocr]'
tesseract --version
tesseract --list-langs
```

No crees el entorno Conda encima de un `.venv` existente: usa un checkout limpio
o un entorno nuevo. Si ya tienes un ejecutable Tesseract confiable, también puedes
configurar `tesseract_cmd` con su ruta sin modificar el sistema. Consulta las
[instrucciones oficiales de instalación](https://tesseract-ocr.github.io/tessdoc/Installation.html)
para tu plataforma. El proyecto no ejecuta gestores de paquetes del sistema.

Para idiomas ausentes, descarga **explícitamente** el archivo requerido del
[repositorio oficial tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)
o [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) después de revisar
su licencia, y guárdalo en una carpeta local, por ejemplo `.venv/share/tessdata`.
Configura `tessdata_dir` con esa ruta. Con `configs/default.yaml` la ruta sería
`../.venv/share/tessdata`. Debe contener `eng.traineddata` para `--language eng`;
`eng+spa` requiere ambos archivos. Nunca se descargan idiomas durante inferencia.

Para CUDA, instala el par torch/torchvision declarado usando el índice compatible
con tus controladores, según el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/),
dentro del entorno antes de instalar el extra detector. CPU es la configuración
predeterminada; `--device mps` o `--device 0` seleccionan Apple Silicon o CUDA.
La disponibilidad y reproducibilidad dependen del hardware y de las wheels.

## Recursos que debes aportar

1. **Imágenes y anotaciones propias o autorizadas**, con transcripciones para
   evaluar OCR. No se requiere descargar un dataset público.
2. **Detector ALPR `.pt` entrenado**, con clase de matrícula conocida y licencia
   comprobada; o **YOLO11n COCO** para ajustar. Obtén `yolo11n.pt` desde los enlaces
   de la [página oficial de YOLO11](https://docs.ultralytics.com/models/yolo11/) y
   guárdalo en `models/yolo11n.pt`. La descarga es manual y explícita. Un detector
   ajustado por este proyecto se guarda en `outputs/training/plates/weights/best.pt`.
3. **Tesseract 5 y datos de idioma** de las fuentes anteriores.

Cuando llegue el detector, la integración básica queda reducida a copiarlo y
validarlo; no hace falta cambiar el código:

```bash
cp /ruta/autorizada/del/modelo.pt models/plate.pt
alpr check-model --config configs/default.yaml --output outputs/model-check.json
```

La copia debe hacerse manualmente desde una fuente conocida. `check-model` no
descarga ni modifica pesos: verifica que el archivo se puede cargar como detector,
lista sus clases, comprueba `class_id` y calcula SHA-256. Completa después
`models/plate.metadata.example.yaml` y conserva el resultado junto al modelo.
Si el proveedor publica una huella, pásala mediante
`--expected-sha256 <sha256>`: se comprobará antes de deserializar los pesos y el
comando abortará si el archivo no coincide.

No distribuimos pesos de terceros. Revisa la licencia de cada dataset y cada peso;
un enlace público no concede automáticamente derechos de uso o redistribución.
Solo carga checkpoints `.pt` confiables: su serialización puede ejecutar código.

## Prueba rápida sin pesos ni conexión

Los fixtures incluidos son carteles `TEST-0`, `TEST-1`, `TEST-2` y un negativo; no
son matrículas fotografiadas ni ejemplos de precisión. La suite sustituye detector
y OCR, e incluye un vídeo artificial generado temporalmente.

```bash
python -m pytest
python -m ruff check src tests
alpr prepare-data --manifest data/synthetic/manifest.json --destination data/demo-yolo --seed 42
```

La última orden crea etiquetas YOLO, particiones y manifiestos utilizables para
comprobar el flujo. No entrenes un modelo destinado a uso real con estos cuatro
ejemplos. Para regenerarlos en otra carpeta:

```bash
alpr synthetic --destination outputs/new-fixtures
```

Las carpetas/archivos de salida deben ser nuevos. Se rechazan colisiones para
evitar sobrescribir datos, pesos o experimentos. Repetir un ejemplo requiere
elegir otra ruta o retirar conscientemente la salida anterior.

## Datos propios y conversión

Anota cada matrícula completa y transcribe sus caracteres. El manifiesto tiene
el mismo formato que `data/synthetic/manifest.json`:

Las transcripciones deben ser no vacías; no mezcles etiquetas desconocidas con
cadenas vacías al evaluar OCR. Para entrenar con matrículas ilegibles puedes
aportar etiquetas YOLO directamente, reservando transcripciones verificadas para
los manifiestos de evaluación de lectura.

```json
[
  {
    "image": "test_0.png",
    "group": "synthetic-0",
    "plates": [{"bbox": [60, 60, 260, 110], "text": "TEST-0"}]
  }
]
```

Este fragmento describe uno de los fixtures existentes; el archivo completo
incluye los cuatro. Para datos propios, `image` apunta a una imagen relativa al
manifiesto; `bbox` usa píxeles `x1,y1,x2,y2`, esquina inferior exclusiva, sobre la
imagen **ya orientada por EXIF**. Una imagen sin matrículas lleva `plates: []`.
Agrupa por vehículo/sesión (sin identificadores personales): no dividas al azar
fotogramas del mismo vídeo. Revisa duplicados de contenido y vehículos repetidos
en distintas sesiones, pues el conversor solo comprueba rutas duplicadas y grupos.

Exporta desde tu anotador a este JSON. Si viene de COCO, transforma
`[x,y,width,height]` a `[x,y,x+width,y+height]`, filtra la categoría de matrícula y
añade `text` y `group`; las transcripciones no se deducen de cajas COCO. También
puedes exportar YOLO directamente con esta estructura:

```text
data/mi-dataset/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
  dataset.yaml
```

Cada imagen tiene un `.txt` de igual nombre (vacío para negativos) con una fila
por matrícula: `0 cx cy w h`, valores normalizados por ancho/alto. El conversor
calcula `cx=(x1+x2)/(2*W)`, `cy=(y1+y2)/(2*H)`, `w=(x2-x1)/W`, `h=(y2-y1)/H`.
La clase única es `0: plate`. Formato y ejemplos oficiales:
[anotaciones YOLO](https://docs.ultralytics.com/datasets/detect/).

Con un manifiesto propio guardado en `data/propios/manifest.json`:

```bash
alpr prepare-data --manifest data/propios/manifest.json --destination data/mi-dataset --seed 42 --train-fraction 0.7 --val-fraction 0.2
```

`data/propios/manifest.json` es una entrada que **debes aportar**, no un archivo
incluido. La orden crea `dataset.yaml`, `split.json` y `train.json`, `val.json`,
`test.json`, además de PNG orientados y etiquetas. Se necesitan al menos tres
grupos. Las fracciones son sobre grupos; su tamaño determina las proporciones de
imágenes. Con pocos grupos se garantizan tres particiones no vacías. Conserva
`test` sin usar durante el ajuste de parámetros. Asegura positivos y negativos
representativos en cada partición: cámaras, países, iluminación, distancias,
reflejos y movimiento. No hay estratificación automática. El entrenamiento y la
evaluación del detector rechazan particiones sin ninguna caja positiva para no
presentar mAP sin referencias; la evaluación de lectura admite imágenes negativas.

Para una exportación manual, `dataset.yaml` debe tener `path` con la raíz local,
`train: images/train`, `val: images/val`, `test: images/test`, `names: {0: plate}`.
La CLI valida imágenes y etiquetas, exige esas carpetas y rechaza instrucciones
`download` en YAML. Es una restricción deliberada para mantener el entrenamiento
local y validable. Si mueves un dataset convertido, actualiza su `path` absoluto.

## Ajustar y evaluar el detector

Tras aportar `models/yolo11n.pt` y crear `data/mi-dataset`:

```bash
alpr train --config configs/train.yaml --data data/mi-dataset/dataset.yaml --epochs 50 --batch 8 --image-size 640 --seed 42
```

Usa `--device 0` para CUDA o `--device mps` si está disponible. Para ajustar de nuevo
un detector ALPR existente usa `--weights models/plate.pt --name plates-v2`.
Se registra `parameters.json` (configuración y versiones), el YAML resuelto,
`metrics.json`, y los archivos de Ultralytics, incluidos `args.yaml`, `results.csv`
y `weights/best.pt` / `last.pt`. Si falla, se conserva `error.json`.

La semilla se fija en Python, NumPy y el entrenador; se solicita determinismo.
Esto no garantiza identidad entre hardware/versiones. Se usa AdamW con tasa
configurable, `workers=0`, sin volteos y `amp=False`; AMP queda desactivado para
evitar verificaciones que pueden intentar descargar modelos auxiliares. Las
gráficas y las integraciones remotas están deshabilitadas. CPU puede ser lento;
reduce lote/resolución para pruebas de funcionamiento y ajusta según tu hardware.

```bash
alpr eval-detector --config configs/default.yaml --weights outputs/training/plates/weights/best.pt --data data/mi-dataset/dataset.yaml --split test --project outputs/evaluation --name detector-test
```

`outputs/evaluation/detector-test/metrics.json` guarda las métricas reales del
framework: precision, recall, mAP@0.5 y mAP@0.5:0.95, entre otras claves nativas.
Esta evaluación usa el barrido de confianza del framework; `confidence` del YAML
es el umbral de **inferencia**, no el umbral para calcular mAP. Ver
[validación oficial](https://docs.ultralytics.com/modes/val/).

`eval-detector` requiere un checkpoint de una sola clase, índice 0, como el que
produce `alpr train`. La inferencia admite checkpoints multiclase mediante
`class_id`, pero para medirlos con este dataset debes adaptarlos primero; se
rechazan en esta orden para evitar métricas con índices de clase incompatibles.

## Inferencia de imagen

Con un detector ALPR autorizado en `models/plate.pt` y Tesseract instalado en el
entorno, este comando usa el fixture incluido como entrada reproducible:

```bash
alpr image data/synthetic/test_0.png --config configs/default.yaml --weights models/plate.pt --language eng --output outputs/image.json --annotated outputs/image.png
```

El detector puede devolver cero cajas sobre un cartel artificial: no es un fallo.
Sustituye la ruta de entrada por una fotografía autorizada para una prueba real.
La CLI admite formatos de imagen de Pillow como JPEG, PNG, BMP, TIFF y WebP,
siempre de una sola imagen/página. Rechaza archivos corruptos o inexistentes,
pesos ausentes y dependencias opcionales ausentes con código de salida 2.
`--annotated` es opcional. También puedes ejecutar `python -m plate_recognition`.

El esquema formal es `schemas/inference.schema.json`. Ejemplo de **estructura**
con ausencia de detecciones, no una medición del detector:

```json
{
  "schema_version": "1.0",
  "media_type": "image",
  "source": "data/synthetic/test_0.png",
  "frames": [{
    "frame_index": null,
    "timestamp_seconds": null,
    "width": 320,
    "height": 160,
    "plates": []
  }]
}
```

Cada objeto de `plates` contiene `bbox`, `text`, `raw_text`,
`detection_confidence`, `ocr_confidence` y `confidence`. Las cajas son enteros
xyxy en la imagen orientada, recortadas a sus límites. `text` usa mayúsculas y
elimina espacios/guiones cuando `normalize: true`; `raw_text` conserva la lectura.
No se impone un patrón de matrícula de un país ni se corrigen caracteres ambiguos.

`ocr_confidence` es la media de confianza de palabras Tesseract ponderada por
caracteres, reescalada de 0–100 a 0–1. `confidence` es su producto por la confianza
del detector. **No es una probabilidad calibrada de acierto** ni una medida de
precisión. OCR vacío conserva la detección, devuelve texto vacío y confianza 0.

## Vídeo

Con un vídeo local autorizado en `data/entrada.mp4` (entrada que debes aportar):

```bash
alpr video data/entrada.mp4 --config configs/default.yaml --weights models/plate.pt --language eng --output outputs/video.json --annotated outputs/video.avi
```

Se procesan todos los frames secuencialmente y se escribe JSON incremental, sin
acumular el vídeo en memoria. La salida usa `media_type: video`, `fps`,
`frame_count` y una entrada por fotograma con `frame_index` desde 0 y
`timestamp_seconds=frame_index/fps`. Las listas vacías se conservan.

El vídeo anotado mantiene dimensiones, orientación y número de frames, con
FFV1 en AVI o `mp4v` en MP4. Se vuelve a decodificar la salida para verificarla.
**No conserva audio ni metadatos originales**. Para vídeo VFR, las marcas de
tiempo son aproximadas y la salida usa FPS constante: convierte a CFR si necesitas
sincronización precisa. La rotación requiere soporte del backend OpenCV; se
rechaza una orientación declarada que el backend no pueda aplicar. Si el backend
no expone los metadatos de orientación, normaliza el archivo previamente.
Dimensiones impares se admiten para JSON; se rechaza su exportación anotada para
evitar el truncado de un píxel de algunos codificadores. No hay tracking,
deduplicación temporal ni voto entre fotogramas. El fallo de decodificación se
detecta por el recuento esperado cuando el contenedor lo proporciona.

## Evaluación de lectura

Se requieren anotaciones con transcripciones correctas. Para aislar OCR, se leen
los recortes definidos por cajas verdaderas, sin necesitar pesos del detector:

```bash
alpr eval-reading --config configs/default.yaml --manifest data/mi-dataset/test.json --mode ocr --output outputs/ocr-test.json
```

Para evaluar también la canalización completa:

```bash
alpr eval-reading --config configs/default.yaml --weights outputs/training/plates/weights/best.pt --manifest data/mi-dataset/test.json --mode e2e --iou 0.5 --output outputs/e2e-test.json
```

- `ocr_ground_truth_crops`: exactitud de matrícula completa y CER de OCR aislado.
- `detection_at_iou`: TP, FP, FN, precision y recall con el umbral de inferencia
  configurado. Emparejamiento uno a uno, predicciones por confianza descendente,
  con la caja real restante de mayor IoU; duplicados cuentan como FP. No es mAP.
- `ocr_matched_detections`: exactitud y CER solo en detecciones emparejadas.
- `end_to_end`: métricas de texto sobre todas las matrículas verdaderas;
  detecciones omitidas producen hipótesis vacías. `exact_plate_recall` exige caja
  emparejada y texto exacto; `exact_plate_precision` penaliza falsos positivos.

CER = suma de distancias de Levenshtein / suma de caracteres de referencia.
Puede superar 1 por inserciones. Exactitud = proporción de cadenas idénticas.
Se aplica la misma normalización a referencia e hipótesis; pon `normalize: false`
para evaluar texto literal. Los denominadores vacíos se expresan como `null`.
Los FP sin referencia no se incluyen en CER; sí en la precisión end-to-end.
La salida no vuelca transcripciones individuales ni imágenes de evaluación.

Para especializar Tesseract en tipografías/alfabetos propios, crea recortes y
transcripciones autorizados, usa modelos `tessdata_best` para el ajuste LSTM con
[tesstrain oficial](https://github.com/tesseract-ocr/tesstrain), exporta el
`traineddata` y selecciona su nombre con `language` y su carpeta con `tessdata_dir`.
Ese entrenamiento OCR requiere su toolchain específico y no se automatiza aquí;
`alpr train` ajusta el detector. También puedes sustituir el adaptador OCR.

## Privacidad, licencias y limitaciones

Todo el procesamiento es local. No se registran imágenes ni transcripciones en
logs; las salidas JSON y las anotaciones se guardan **solo cuando se solicitan**.
Los JSON contienen texto y rutas locales, por lo que pueden ser sensibles. Evita
incluir datos personales en rutas/grupos, controla accesos y define retención.

No hay descarga automática de pesos o datos. Se exigen rutas locales, se activa
`YOLO_OFFLINE`, se desactiva la instalación automática, telemetría e integraciones
de Ultralytics, y se bloquean conexiones/DNS de Python durante sus llamadas.
Las cachés de motores se ubican en el prefijo del entorno virtual salvo variables
de entorno explícitas. El bloqueo está diseñado para esta CLI secuencial; no es
un sandbox para checkpoints maliciosos ni un servicio concurrente. En un entorno
que exija aislamiento garantizado, añade controles de red a nivel del sistema.

Comprueba licencias del software, pesos, imágenes y anotaciones antes de uso o
redistribución. Ultralytics ofrece condiciones AGPL-3.0/Enterprise: consulta la
[licencia oficial](https://www.ultralytics.com/license). Tesseract usa Apache-2.0;
los datos de idioma y otros recursos deben revisarse por separado. No se incluye
una licencia de redistribución para tus datos ni una garantía de cumplimiento.
Imágenes de vehículos y matrículas pueden ser datos personales: verifica la base
legal, permisos de captura, finalidad, minimización, retención y normativa de tu
jurisdicción, incluido RGPD cuando proceda.

Reflejos, baja resolución, perspectiva, suciedad, movimiento, exposición y
matrículas de dos líneas o de otros países pueden degradar ambos componentes.
El recorte rectangular no rectifica perspectiva ni corrige inclinaciones. Los
idiomas OCR genéricos no conocen todas las tipografías de matrícula. La fuente
incluida en las anotaciones cubre principalmente latín; JSON conserva Unicode.
Este proyecto necesita validación de datos, licencias, latencia, precisión y
sesgos en el entorno objetivo antes de plantear producción.

## Problemas frecuentes

- **Faltan pesos:** revisa `--weights`. No se acepta un nombre remoto para que
  Ultralytics lo descargue. Un checkpoint COCO no detecta matrículas por defecto.
- **Falta Tesseract:** instala el ejecutable en tu entorno dedicado, comprueba
  `tesseract --version` y configura `tesseract_cmd` si no está en PATH.
- **Falta idioma:** revisa `tesseract --list-langs` y los `.traineddata` locales.
  Los códigos de Tesseract son `eng`, `spa`, etc., no `en`/`es`.
- **Sin detecciones:** comprueba la clase, el checkpoint y el dominio de los datos.
  Ajusta `confidence` en validación; no deduzcas precisión de una sola imagen.
- **OCR deficiente:** aumenta resolución de captura, prueba `binarize: true`,
  `ocr_height` y PSM 6/7/8 en validación. Aumentar un recorte no recupera detalle.
- **CUDA/MPS o memoria:** prueba `--device cpu`, baja `--batch`/`--image-size`.
  Verifica versiones torch/torchvision y controladores con la guía oficial.
- **Códec no disponible:** usa `.avi`/FFV1 o `.mp4`/mp4v según tu build OpenCV;
  puedes guardar solo JSON. No se modifica silenciosamente el tamaño.
- **Salida existente:** usa una ruta nueva o un `--name` nuevo. JSON se publica
  mediante un temporal solo al finalizar; elimina conscientemente salidas previas.
- **Error durante entrenamiento:** consulta `error.json` y `parameters.json` del
  experimento. No añadas `download` a los YAML; prepara recursos explícitamente.

## Fuentes técnicas

- [Ultralytics: entrenamiento](https://docs.ultralytics.com/modes/train/),
  [inferencia](https://docs.ultralytics.com/modes/predict/),
  [versión utilizada](https://pypi.org/project/ultralytics/8.3.221/).
- [PyTorch y torchvision: versiones compatibles](https://pytorch.org/get-started/previous-versions/).
- [pytesseract: API, idiomas y confianza TSV](https://github.com/madmaze/pytesseract),
  [Tesseract: uso CLI](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).
- [OpenCV: distribución Python](https://pypi.org/project/opencv-python/4.12.0.88/),
  [propiedades de vídeo y orientación](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html).
