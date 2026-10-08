#!/usr/bin/env python3
"""Grafica el historial producido por red_neuronal.py.

Uso habitual desde la carpeta del proyecto:
    python graficar_aprendizaje.py

El historial se conserva en CSV para poder abrirlo también con una hoja de
cálculo o reutilizarlo en otros análisis, sin tener que volver a entrenar.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Al resolver la ruta desde el propio archivo, el script funciona aunque se
# ejecute desde otra carpeta (por ejemplo: python /ruta/graficar_aprendizaje.py).
CARPETA_SCRIPT = Path(__file__).resolve().parent
HISTORIAL_POR_DEFECTO = CARPETA_SCRIPT / "historial_entrenamiento.csv"
SALIDA_POR_DEFECTO = CARPETA_SCRIPT

# Estas columnas forman el pequeño contrato entre el entrenamiento y este
# visualizador. El CSV incluye una fila por época.
COLUMNAS_NECESARIAS = [
    "epoca",
    "train_mse",
    "validation_mse",
    "learning_rate",
]


def leer_argumentos():
    """Permite elegir otro historial o una carpeta de salida sin editar código."""
    parser = argparse.ArgumentParser(
        description="Grafica las métricas de aprendizaje de la red neuronal."
    )
    parser.add_argument(
        "--historial",
        type=Path,
        default=HISTORIAL_POR_DEFECTO,
        help="CSV creado por red_neuronal.py (por defecto: junto a este script).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=SALIDA_POR_DEFECTO,
        help="Carpeta donde guardar las figuras PNG y PDF.",
    )
    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="Además de guardar las figuras, abrirlas en una ventana interactiva.",
    )
    return parser.parse_args()


def cargar_historial(ruta_csv):
    """Lee el CSV y comprueba que se pueda usar para construir las gráficas."""
    if not ruta_csv.is_file():
        raise SystemExit(
            f"No se encontró el historial: {ruta_csv}\n"
            "Ejecuta primero red_neuronal.py para entrenar y guardar las métricas."
        )

    historial = pd.read_csv(ruta_csv)

    if historial.empty:
        raise SystemExit(f"El historial no contiene épocas: {ruta_csv}")

    faltantes = [
        columna for columna in COLUMNAS_NECESARIAS if columna not in historial.columns
    ]
    if faltantes:
        raise SystemExit(
            "Al historial le faltan estas columnas: " + ", ".join(faltantes)
        )

    # Convertir explícitamente detecta celdas de texto o valores dañados antes
    # de que Matplotlib falle con un error menos claro al intentar graficarlos.
    for columna in COLUMNAS_NECESARIAS:
        historial[columna] = pd.to_numeric(historial[columna], errors="raise")

    # Ordenar evita que la línea retroceda si el CSV fue reordenado manualmente.
    return historial.sort_values("epoca").reset_index(drop=True)


def crear_grafica(historial, carpeta_salida, mostrar=False):
    """Dibuja las métricas y guarda una versión de imagen y otra vectorial."""
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    epocas = historial["epoca"]
    mejor_indice = historial["validation_mse"].idxmin()
    mejor_epoca = int(historial.loc[mejor_indice, "epoca"])
    mejor_mse_validacion = historial.loc[mejor_indice, "validation_mse"]

    # Tres paneles ayudan a responder preguntas distintas:
    # 1) ¿Está bajando el error? 2) ¿Se separan train y validation?
    # 3) ¿Qué tasa de aprendizaje se utilizó en cada época?
    figura, (eje_perdida, eje_brecha, eje_tasa) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.3, 1]},
    )
    figura.suptitle("Evolución del aprendizaje de la red neuronal", fontsize=15)

    # La MSE está expresada en pedidos al cuadrado. El eje logarítmico permite
    # apreciar tanto la caída inicial grande como mejoras más pequeñas al final.
    eje_perdida.plot(
        epocas,
        historial["train_mse"],
        color="#021093",
        linewidth=2,
        label="Entrenamiento",
    )
    eje_perdida.plot(
        epocas,
        historial["validation_mse"],
        color="#90302D",
        linewidth=2,
        label="Validación",
    )
    eje_perdida.scatter(
        [mejor_epoca],
        [mejor_mse_validacion],
        color="#168C75",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label=f"Mejor validación: época {mejor_epoca}",
    )
    eje_perdida.axvline(
        mejor_epoca, color="#168C75", linestyle="--", alpha=0.6
    )
    if (historial[["train_mse", "validation_mse"]] > 0).all().all():
        eje_perdida.set_yscale("log")
        eje_perdida.set_ylabel("MSE (escala logarítmica)")
    else:
        # La escala lineal evita problemas si una configuración produce MSE 0.
        eje_perdida.set_ylabel("MSE (pedidos al cuadrado)")
    eje_perdida.set_title("Pérdida por época: menor suele ser mejor")
    eje_perdida.grid(True, linestyle=":", alpha=0.55)
    eje_perdida.legend()

    # La brecha es validación menos entrenamiento. Si crece y permanece
    # positiva, la red mejora en los datos vistos pero generaliza peor: posible
    # sobreajuste. Una brecha pequeña no demuestra por sí sola que el modelo sea
    # bueno; también hay que mirar el nivel absoluto de ambas pérdidas.
    brecha = historial["validation_mse"] - historial["train_mse"]
    eje_brecha.plot(epocas, brecha, color="#7A5195", linewidth=1.8)
    eje_brecha.axhline(0, color="#444444", linewidth=1, linestyle="--")
    eje_brecha.fill_between(epocas, brecha, 0, where=(brecha >= 0),
                            color="#D9534F", alpha=0.18, interpolate=True)
    eje_brecha.fill_between(epocas, brecha, 0, where=(brecha < 0),
                            color="#2878B5", alpha=0.14, interpolate=True)
    eje_brecha.set_ylabel("Validación - entrenamiento")
    eje_brecha.set_title("Brecha de generalización: vigilar si aumenta")
    eje_brecha.grid(True, linestyle=":", alpha=0.55)

    # Aquí se verá una línea plana si la tasa fue constante (como en la
    # configuración actual) y una curva si más adelante se usa un scheduler.
    eje_tasa.plot(
        epocas,
        historial["learning_rate"],
        color="#E18A24",
        linewidth=1.8,
        marker="o" if len(historial) <= 30 else None,
        markersize=3,
    )
    eje_tasa.set_ylabel("Tasa de aprendizaje")
    eje_tasa.set_xlabel("Época")
    eje_tasa.set_title("Tasa de aprendizaje aplicada")
    eje_tasa.grid(True, linestyle=":", alpha=0.55)

    # tight_layout reserva espacio para el título general y evita que las
    # etiquetas de los paneles se superpongan.
    figura.tight_layout(rect=(0, 0, 1, 0.97))

    ruta_png = carpeta_salida / "curva_aprendizaje.png"
    ruta_pdf = carpeta_salida / "curva_aprendizaje.pdf"
    figura.savefig(ruta_png, dpi=180, bbox_inches="tight")
    figura.savefig(ruta_pdf, bbox_inches="tight")

    print(f"Épocas graficadas: {len(historial)}")
    print(f"Mejor época según validación: {mejor_epoca}")
    print(f"MSE mínima de validación: {mejor_mse_validacion:,.2f}")
    print(f"Imagen guardada en: {ruta_png}")
    print(f"Documento PDF guardado en: {ruta_pdf}")

    if mostrar:
        # Útil en un escritorio; en servidores sin interfaz gráfica basta con
        # las imágenes guardadas, por eso mostrar la ventana es opcional.
        plt.show()

    plt.close(figura)


def main():
    """Coordina lectura, validación y creación de las gráficas."""
    argumentos = leer_argumentos()
    historial = cargar_historial(argumentos.historial)
    crear_grafica(historial, argumentos.salida, argumentos.mostrar)


if __name__ == "__main__":
    main()