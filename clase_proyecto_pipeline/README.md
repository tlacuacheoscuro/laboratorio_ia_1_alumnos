# El Pipeline Completo de IA (Banderazo del Proyecto Final)

Clase de arranque y estructuración del proyecto final del semestre (10 de noviembre al 15 de enero). Conecta las herramientas vistas en el curso dentro de un pipeline reproducible de 6 etapas.

## Archivos

- `generar_datos_ejemplo.py`: crea `datos_ejemplo_pipeline.csv` (180 días sintéticos con un faltante, un valor físicamente imposible y dos valores atípicos extremos).
- `pipeline_base.py`: plantilla modular en Python estándar (sin dependencias externas) que ejecuta las 6 etapas del pipeline y exporta `predicciones_prueba.csv`.
- `plantilla_propuesta.md`: acta de inicio del proyecto final para que cada equipo defina hoy su ruta (A, B o C), su pregunta, sus datos, su referencia (*baseline*) y sus 4 hitos hacia el 15 de enero.
- `diapositivas_proyecto_pipeline.tex` / `diapositivas_proyecto_pipeline.pdf`: presentación Beamer (13 diapositivas).
- `guia_docente_proyecto_pipeline.md`: plan de clase de 75 minutos, calendario de dos meses, rúbrica y preguntas filtro para asesorar a los equipos.

## Ejecución

```bash
cd clase_proyecto_pipeline
python3 generar_datos_ejemplo.py
python3 pipeline_base.py
pdflatex diapositivas_proyecto_pipeline.tex
```

