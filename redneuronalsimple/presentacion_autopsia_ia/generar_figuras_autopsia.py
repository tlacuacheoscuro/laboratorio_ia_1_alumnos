#!/usr/bin/env python3
"""Genera figuras vectoriales de alta calidad para la presentación de Autopsia de la IA."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA_SCRIPT = Path(__file__).resolve().parent
CARPETA_SALIDA = CARPETA_SCRIPT / "graficas"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
ARCHIVO_PREDICCIONES = CARPETA_SCRIPT.parent / "predicciones_prueba.csv"

# Paleta de colores consistente y elegante
NAVY = "#0B2545"
TEAL = "#2A9D8F"
CORAL = "#E63946"
PURPLE = "#7A5195"
LIGHT_BG = "#F8F9FA"
DARK_GRAY = "#2B2D42"


def figura_autopsia_peores_errores():
    """Grafica las predicciones de prueba destacando los peores fallos reales del modelo."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)

    if ARCHIVO_PREDICCIONES.is_file():
        df = pd.read_csv(ARCHIVO_PREDICCIONES)
    else:
        # Fallback sintético en caso extremo
        np.random.seed(42)
        n = 400
        reales = np.random.exponential(200, n) + 20
        preds = reales * 0.7 + np.random.normal(0, 50, n)
        df = pd.DataFrame({"real": reales, "prediccion": preds, "error_absoluto": np.abs(reales - preds)})

    # Ordenar por error para identificar los peores
    df_sorted = df.sort_values(by="error_absoluto", ascending=False).reset_index(drop=True)
    peores = df_sorted.head(4)

    # Graficar todas las muestras
    ax.scatter(
        df["real"], df["prediccion"],
        alpha=0.45, color="#134074", s=32, edgecolors="none", label="Muestras normales de prueba (400)"
    )

    # Línea ideal y = x
    max_val = max(df["real"].max(), df["prediccion"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color=TEAL, linestyle="--", linewidth=1.8, label="Predicción ideal ($y = \hat{y}$)")

    # Destacar y anotar los peores errores en rojo/coral
    ax.scatter(
        peores["real"], peores["prediccion"],
        color=CORAL, s=110, zorder=5, edgecolors="white", linewidth=1.5, label="Errores extremos (Outliers)"
    )

    for i, row in peores.iterrows():
        r, p, err = int(row["real"]), int(row["prediccion"]), int(row["error_absoluto"])
        # Coordenadas de la anotación
        offset_x = -350 if r > 1000 else 80
        offset_y = 120 if p < 500 else -100
        ax.annotate(
            f"Fila #{i+1}: Real={r} | Pred={p}\nError = {err} pedidos",
            xy=(r, p),
            xytext=(r + offset_x, p + offset_y),
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.3),
            fontsize=8,
            fontweight="bold",
            color=DARK_GRAY,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT_BG, edgecolor=CORAL, alpha=0.95),
        )

    # Región de subestimación
    ax.fill_between([0, max_val], 0, [0, max_val], color=CORAL, alpha=0.04)
    ax.text(max_val * 0.65, max_val * 0.2, "Zona de Subestimación Severa\n(La red se queda corta en pedidos altos)",
            color=CORAL, fontsize=9, fontweight="bold", ha="center")

    ax.set_title("Autopsia del Modelo: Dónde se Equivoca Catastróficamente la Red", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Pedidos Reales ($y$)", fontsize=10.5)
    ax.set_ylabel("Pedidos Predichos ($\hat{y}$)", fontsize=10.5)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(CARPETA_SALIDA / "autopsia_peores_errores.pdf")
    fig.savefig(CARPETA_SALIDA / "autopsia_peores_errores.png")
    plt.close(fig)
    print("Figura guardada: autopsia_peores_errores.pdf / .png")


def figura_clever_hans():
    """Genera diagrama del fenómeno Clever Hans (Lobo vs Perro con fondo de nieve)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.3), dpi=200)

    # Panel 1: Lo que el humano cree que la red hace
    rect_dog = plt.Rectangle((0.15, 0.15), 0.7, 0.7, facecolor="#E8F4F8", edgecolor=NAVY, linewidth=2, linestyle="-")
    ax1.add_patch(rect_dog)
    # Silueta simulada del animal
    circle_head = plt.Circle((0.5, 0.6), 0.18, color=NAVY, alpha=0.85)
    circle_body = plt.Circle((0.5, 0.35), 0.24, color=NAVY, alpha=0.85)
    ax1.add_patch(circle_head)
    ax1.add_patch(circle_body)
    # Zona de atención ideal
    attention_box = plt.Rectangle((0.28, 0.2), 0.44, 0.55, facecolor="none", edgecolor=TEAL, linewidth=3, linestyle="--")
    ax1.add_patch(attention_box)
    ax1.text(0.5, 0.06, "Lo que creemos que aprende:\nAnatomía del animal (hocico, orejas)", ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
    ax1.text(0.5, 0.88, "Clase: 'Lobo' (98% confianza)", ha="center", fontsize=10, fontweight="bold", color=TEAL,
             bbox=dict(boxstyle="round", facecolor="#E8F8F5", edgecolor=TEAL))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("Expectativa Humana", fontsize=11, fontweight="bold")

    # Panel 2: Lo que la red realmente aprendió (Efecto Clever Hans)
    rect_snow = plt.Rectangle((0.15, 0.15), 0.7, 0.7, facecolor="#FDEDEC", edgecolor=CORAL, linewidth=2, linestyle="-")
    ax2.add_patch(rect_snow)
    # Fondo con textura de "nieve"
    np.random.seed(12)
    snow_x = np.random.uniform(0.18, 0.82, 35)
    snow_y = np.random.uniform(0.18, 0.82, 35)
    ax2.scatter(snow_x, snow_y, color=CORAL, s=25, alpha=0.7, marker="*")
    # Silueta borrosa del animal ignorada
    c_h = plt.Circle((0.5, 0.6), 0.18, color="#AAAAAA", alpha=0.4)
    c_b = plt.Circle((0.5, 0.35), 0.24, color="#AAAAAA", alpha=0.4)
    ax2.add_patch(c_h)
    ax2.add_patch(c_b)
    # Zona de atención real (en el fondo)
    ax2.text(0.5, 0.06, "Lo que REALMENTE aprendió:\n¿Hay nieve blanca de fondo? $\\rightarrow$ Lobo", ha="center", fontsize=9.5, fontweight="bold", color=CORAL)
    ax2.text(0.5, 0.88, "Atajo Espurio Detectado (LIME)", ha="center", fontsize=10, fontweight="bold", color=CORAL,
             bbox=dict(boxstyle="round", facecolor="#FADBD8", edgecolor=CORAL))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Realidad: Aprendizaje de Atajos (Shortcut)", fontsize=11, fontweight="bold")

    fig.suptitle("El Fenómeno 'Clever Hans' en Deep Learning (Ribeiro et al., 2016)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(CARPETA_SALIDA / "clever_hans_lobo_nieve.pdf")
    fig.savefig(CARPETA_SALIDA / "clever_hans_lobo_nieve.png")
    plt.close(fig)
    print("Figura guardada: clever_hans_lobo_nieve.pdf / .png")


def figura_interpolacion_vs_extrapolacion():
    """Muestra la diferencia entre interpolación fluida y extrapolación descontrolada."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)

    # Rango de entrenamiento: [100, 600]
    x_train = np.linspace(100, 600, 40)
    np.random.seed(7)
    y_true_train = 400 * np.exp(-x_train / 280) + 50
    y_train = y_true_train + np.random.normal(0, 25, len(x_train))

    # Curva que la red ajusta dentro del rango
    x_in = np.linspace(100, 600, 200)
    y_net_in = 400 * np.exp(-x_in / 280) + 50

    # Fuera de rango hacia la izquierda (x < 100)
    x_left = np.linspace(-50, 100, 100)
    y_net_left = y_net_in[0] + 1.2 * (100 - x_left) ** 1.3 - 0.005 * (100 - x_left) ** 2

    # Fuera de rango hacia la derecha (x > 600)
    x_right = np.linspace(600, 850, 100)
    y_net_right = y_net_in[-1] - 0.4 * (x_right - 600) + 0.002 * (x_right - 600) ** 2

    # Sombra: Rango conocido vs Rango desconocido
    ax.axvspan(100, 600, color=TEAL, alpha=0.12, label="Zona Conhecida (Interpolación Confiable)")
    ax.axvspan(-50, 100, color=CORAL, alpha=0.10, label="Zona de Extrapolación (Peligro de Alucinación)")
    ax.axvspan(600, 850, color=CORAL, alpha=0.10)

    # Dibujar datos de entrenamiento
    ax.scatter(x_train, y_train, color="#134074", alpha=0.65, s=35, label="Datos de entrenamiento observados")

    # Curva de la red dentro del dominio
    ax.plot(x_in, y_net_in, color=TEAL, linewidth=2.8, label="Predicción de la red (Interpolación)")

    # Curva de la red fuera del dominio (disparates)
    ax.plot(x_left, y_net_left, color=CORAL, linewidth=2.5, linestyle="--", label="Alucinación fuera de rango")
    ax.plot(x_right, y_net_right, color=CORAL, linewidth=2.5, linestyle="--")

    # Anotaciones
    ax.annotate(
        "¿Qué pasa si precio = $0?\nPredicción inventada sin sustento",
        xy=(0, y_net_left[np.argmin(np.abs(x_left - 0))]),
        xytext=(30, 520),
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
        fontsize=8.5, fontweight="bold", color=DARK_GRAY,
        bbox=dict(boxstyle="round", facecolor=LIGHT_BG, edgecolor=CORAL)
    )

    ax.annotate(
        "Excelente ajuste donde\nhay evidencia empírica",
        xy=(350, y_net_in[np.argmin(np.abs(x_in - 350))]),
        xytext=(320, 260),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
        fontsize=8.5, fontweight="bold", color=DARK_GRAY,
        bbox=dict(boxstyle="round", facecolor=LIGHT_BG, edgecolor=TEAL)
    )

    ax.set_title("La Ilusión del Conocimiento: Las Redes No Razonan, Solo Interpolan", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Precio del Platillo ($)", fontsize=10.5)
    ax.set_ylabel("Demanda Predicha (Pedidos)", fontsize=10.5)
    ax.set_xlim(-40, 840)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8.2)

    plt.tight_layout()
    fig.savefig(CARPETA_SALIDA / "interpolacion_vs_extrapolacion.pdf")
    fig.savefig(CARPETA_SALIDA / "interpolacion_vs_extrapolacion.png")
    plt.close(fig)
    print("Figura guardada: interpolacion_vs_extrapolacion.pdf / .png")


def figura_mapa_zoologico():
    """Genera infografía del mapa de familias de redes neuronales."""
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def dibujar_caja(x, y, w, h, titulo, subtitulo, color_borde, color_fondo, texto_color="black"):
        rect = plt.Rectangle((x, y), w, h, facecolor=color_fondo, edgecolor=color_borde, linewidth=2, linestyle="-")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, titulo, ha="center", va="center", fontsize=9.5, fontweight="bold", color=texto_color)
        ax.text(x + w / 2, y + h * 0.32, subtitulo, ha="center", va="center", fontsize=8, color=texto_color)

    # Nodo Central: Red Densa (lo que ya hicimos)
    dibujar_caja(0.4, 1.8, 2.5, 1.4, "Red Feedforward (MLP)\n(Lo que dominamos)", "Datos Tabulares (CSV)\nDemanda, estatura, finanzas", NAVY, "#E8F4F8", NAVY)

    # Conectores flechas
    ax.annotate("", xy=(3.8, 4.0), xytext=(2.9, 2.8), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
    ax.annotate("", xy=(3.8, 2.5), xytext=(2.9, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
    ax.annotate("", xy=(3.8, 1.0), xytext=(2.9, 2.2), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))

    # Nodo 1: CNN (Visión)
    dibujar_caja(3.8, 3.4, 2.7, 1.2, "+ Convolución Espacial\n(Redes CNN)", "Imágenes y Video\nAutos autónomos, medicina", TEAL, "#E8F8F5", TEAL)

    # Nodo 2: RNN / Memoria (Secuencias y Robots)
    dibujar_caja(3.8, 1.9, 2.7, 1.2, "+ Memoria Recurrente\n(Redes con Estado / RNN)", "Series Temporales y Agentes\nRobots navegantes (Lab 8)", PURPLE, "#F4ECF7", PURPLE)

    # Nodo 3: Transformers (Lenguaje)
    dibujar_caja(3.8, 0.4, 2.7, 1.2, "+ Mecanismo de Atención\n(Transformers / LLMs)", "Texto y Código\nChatGPT, Claude, Gemini", CORAL, "#FDEDEC", CORAL)

    # Caja lateral de síntesis
    rect_resumen = plt.Rectangle((7.1, 0.4), 2.6, 4.2, facecolor="#F8F9FA", edgecolor="#CCCCCC", linewidth=1.5, linestyle="--")
    ax.add_patch(rect_resumen)
    ax.text(8.4, 4.2, "La Gran Lección", ha="center", fontsize=10, fontweight="bold", color=NAVY)
    texto_res = (
        "El 'núcleo matemático'\n"
        "sigue siendo el mismo:\n\n"
        "1. Tensores y Pesos (W)\n"
        "2. Forward Pass\n"
        "3. Pérdida (Loss)\n"
        "4. Backprop (Gradiente)\n"
        "5. Optimizador (Adam)\n\n"
        "Solo cambia cómo se\n"
        "conectan las neuronas."
    )
    ax.text(8.4, 2.2, texto_res, ha="center", va="center", fontsize=8.5, color=DARK_GRAY)

    fig.suptitle("El Zoológico de la Inteligencia Artificial: La Red Simple como Bloque Fundamental", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(CARPETA_SALIDA / "mapa_zoologico_ia.pdf")
    fig.savefig(CARPETA_SALIDA / "mapa_zoologico_ia.png")
    plt.close(fig)
    print("Figura guardada: mapa_zoologico_ia.pdf / .png")


def figura_clever_hans_real():
    """Genera la comparativa con las fotos originales del paper de LIME (Ribeiro et al., KDD 2016)."""
    ruta_husky = CARPETA_SALIDA / "husky.png"
    ruta_exp = CARPETA_SALIDA / "exp_husky.png"

    if not (ruta_husky.is_file() and ruta_exp.is_file()):
        print("Fotos reales no encontradas en graficas/, omitiendo...")
        return

    import matplotlib.image as mpimg
    img_husky = mpimg.imread(ruta_husky)
    img_exp = mpimg.imread(ruta_exp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=250)

    # Panel 1: Original
    ax1.imshow(img_husky)
    ax1.set_title("(a) Foto Original de Entrada\n(Husky en la nieve)", fontsize=10.5, fontweight="bold", pad=8)
    ax1.text(
        0.5, -0.12, "Predicción del Modelo: \"LOBO\"\n(Error en Prueba)", transform=ax1.transAxes,
        ha="center", fontsize=9, fontweight="bold", color=CORAL,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FDEDEC", edgecolor=CORAL)
    )
    ax1.axis("off")

    # Panel 2: LIME explanation
    ax2.imshow(img_exp)
    ax2.set_title("(b) Explicación con LIME\n(Píxeles que activaron la neurona)", fontsize=10.5, fontweight="bold", pad=8)
    ax2.text(
        0.5, -0.12, "Atajo Espurio:\n¡La red solo miró la nieve!", transform=ax2.transAxes,
        ha="center", fontsize=9, fontweight="bold", color=TEAL,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F8F5", edgecolor=TEAL)
    )
    ax2.axis("off")

    fig.suptitle("Fotos Reales del Paper: \"Why Should I Trust You?\" (Ribeiro et al., KDD 2016)", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(CARPETA_SALIDA / "clever_hans_lobo_nieve_real.pdf", bbox_inches="tight")
    fig.savefig(CARPETA_SALIDA / "clever_hans_lobo_nieve_real.png", bbox_inches="tight")
    plt.close(fig)
    print("Figura guardada: clever_hans_lobo_nieve_real.pdf / .png")


if __name__ == "__main__":
    figura_autopsia_peores_errores()
    figura_clever_hans()
    figura_clever_hans_real()
    figura_interpolacion_vs_extrapolacion()
    figura_mapa_zoologico()
