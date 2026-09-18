# Clase 5 — Introducción a Redes Neuronales

Archivos añadidos:

- `clase5_presentacion.tex` — presentación Beamer.
- `data/generate_height_data.py` — script para generar el CSV sintético.
- `data/estatura_ninos.csv` — (generado a continuación) dataset sintético.
- `scripts/train_estatura.py` — script PyTorch para entrenar un modelo sencillo.

Instrucciones rápidas:

1. Generar datos (si quieres cambiar cantidad o semilla):

```bash
python3 data/generate_height_data.py --n 500 --out data/estatura_ninos.csv --seed 42
```

2. Instalar dependencias (recomiendo crear un virtualenv):

```bash
pip install pandas numpy torch
```

3. Entrenar el modelo:

```bash
python3 scripts/train_estatura.py --data data/estatura_ninos.csv --epochs 300
```

4. Compilar la presentación (pdflatex / lualatex / xelatex):

```bash
pdflatex clase5_presentacion.tex
```
