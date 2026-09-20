# Recursos locales

Esta carpeta no contiene pesos. Para inferencia aporta un detector YOLO `.pt`
entrenado para matrículas y configura `weights`/`--weights` y `class_id`.
Para ajustar desde COCO descarga manualmente `yolo11n.pt` desde los enlaces
oficiales de https://docs.ultralytics.com/models/yolo11/ y revisa su licencia.
COCO no contiene una clase de matrícula: no uses ese checkpoint para evaluar ALPR
sin ajustarlo. Los `.pt` deben proceder de una fuente confiable porque pueden
contener código serializado. Los pesos están excluidos de Git.

Tesseract necesita el ejecutable y `eng.traineddata` (u otro idioma). Instálalos
explícitamente siguiendo https://tesseract-ocr.github.io/tessdoc/Installation.html
o aporta `tessdata_dir` con archivos de https://github.com/tesseract-ocr/tessdata_fast
y revisa la licencia del repositorio/archivo concreto. No hay descarga en inferencia.

