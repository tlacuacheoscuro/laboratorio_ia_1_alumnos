# Del CSV al modelo

Clase integradora de análisis tabular, valores atípicos y regresión supervisada, basada en ventas sintéticas reproducibles.

## Archivos

- `generar_ventas.py`: crea `ventas_clase.csv` con 240 filas.
- `analizar_ventas.py`: inspecciona, limpia, resume, marca atípicos y evalúa una regresión.
- `diapositivas_csv_ia.tex` / `diapositivas_csv_ia.pdf`: presentación Beamer.
- `guia_docente_csv_ia.md`: objetivos, plan de 75 minutos y tarea.

## Ejecución

```bash
python3 generar_ventas.py
python3 analizar_ventas.py
pdflatex diapositivas_csv_ia.tex
```

El análisis requiere NumPy. La generación y lectura del CSV usan la biblioteca estándar. El conjunto es didáctico, no representa ventas reales; los valores atípicos se señalan para investigarlos y no se borran automáticamente.