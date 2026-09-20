# Datos propios

No se distribuyen fotografías de vehículos ni matrículas reales. `synthetic/`
contiene carteles geométricos TEST dibujados por el generador del proyecto, y
un fondo negativo. Sirven para comprobar contratos de software, nunca para
afirmar precisión o entrenar un detector útil.

El manifiesto es una lista JSON. `image` es relativa al manifiesto; `group`
identifica el vehículo o sesión y evita repartir fotogramas correlacionados entre
particiones. `plates` contiene `bbox: [x1, y1, x2, y2]` en píxeles (extremo inferior
exclusivo) y `text` con la transcripción. Las coordenadas se refieren a la imagen
orientada por EXIF. Una imagen negativa tiene `plates: []`.

`alpr prepare-data` valida todas las entradas, divide grupos con semilla fija,
crea PNG orientados y etiquetas YOLO, y guarda `dataset.yaml`, `split.json` y los
manifiestos `train.json`, `val.json`, `test.json`. Las fracciones son de grupos,
no de imágenes; con pocos grupos las proporciones se ajustan para que no haya
particiones vacías. Revisa la distribución de países, cámaras y condiciones.
Los identificadores de grupo deben ser seudónimos; no uses información personal.

