#!/usr/bin/env python3
"""Genera figuras conceptuales para la presentación de Early Stopping."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CARPETA_SALIDA = Path(__file__).resolve().parent / "graficas"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Paleta de colores armoniosa
NAVY = "#0B2545"
TEAL = "#2A9D8F"
CORAL = "#E63946"
PURPLE = "#7A5195"
GRAY = "#6C757D"
LIGHT_BG = "#F8F9FA"


def generar_esquema_early_stopping():
    """Genera diagrama conceptual de curvas de pérdida y parada temprana."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)

    epocas = np.linspace(1, 60, 300)

    # Curva de entrenamiento: decae continuamente (asintótica hacia cero)
    loss_train = 1.2 * np.exp(-epocas / 12) + 0.15 * np.exp(-epocas / 40) + 0.05

    # Curva de validación: baja hasta un mínimo (época ~20) y luego sube por overfitting
    mejor_ep = 20.0
    loss_val = (
        1.3 * np.exp(-epocas / 10)
        + 0.00045 * (epocas - mejor_ep) ** 2
        + 0.18
        + 0.02 * np.sin(epocas * 0.8) * np.exp(-epocas / 25)
    )

    # Punto óptimo
    idx_opt = np.argmin(loss_val)
    ep_opt = epocas[idx_opt]
    val_opt = loss_val[idx_opt]

    # Punto de parada temprana (15 épocas después)
    ep_stop = ep_opt + 15.0
    idx_stop = np.argmin(np.abs(epocas - ep_stop))
    val_stop = loss_val[idx_stop]

    # Dibujar curvas
    ax.plot(epocas, loss_train, color="#134074", linewidth=2.5, label="Pérdida Entrenamiento ($J_{train}$)")
    ax.plot(epocas, loss_val, color=PURPLE, linewidth=2.5, label="Pérdida Validación ($J_{val}$)")

    # Sombrear zonas: Subajuste, Generalización Óptima, Sobreajuste
    ax.axvspan(1, ep_opt - 5, color="#2878B5", alpha=0.08, label="Zona de Subajuste (Underfitting)")
    ax.axvspan(ep_opt - 5, ep_opt + 5, color=TEAL, alpha=0.12, label="Ventana Óptima de Generalización")
    ax.axvspan(ep_opt + 5, 60, color=CORAL, alpha=0.08, label="Zona de Sobreajuste (Overfitting)")

    # Marcar mejor época (Checkpoint)
    ax.scatter([ep_opt], [val_opt], color=TEAL, s=120, zorder=5, edgecolors="white", linewidth=1.5)
    ax.axvline(ep_opt, color=TEAL, linestyle="--", linewidth=1.8, alpha=0.9)
    ax.annotate(
        f"Mejor Checkpoint (θ*)\nÉpoca óptima (Guardado en disco)",
        xy=(ep_opt, val_opt),
        xytext=(ep_opt - 12, val_opt + 0.35),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
        fontsize=9,
        fontweight="bold",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_BG, edgecolor=TEAL, alpha=0.95),
    )

    # Marcar parada temprana
    ax.scatter([ep_stop], [val_stop], color=CORAL, s=120, zorder=5, edgecolors="white", linewidth=1.5)
    ax.axvline(ep_stop, color=CORAL, linestyle="-.", linewidth=1.8, alpha=0.9)
    ax.annotate(
        f"Early Stopping Activado\n(Paciencia = 15 épocas sin mejora)",
        xy=(ep_stop, val_stop),
        xytext=(ep_stop + 3, val_stop + 0.25),
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.5),
        fontsize=9,
        fontweight="bold",
        color=CORAL,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_BG, edgecolor=CORAL, alpha=0.95),
    )

    # Flecha de paciencia entre ep_opt y ep_stop
    ax.annotate(
        "",
        xy=(ep_stop, 0.95),
        xytext=(ep_opt, 0.95),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5),
    )
    ax.text((ep_opt + ep_stop) / 2, 0.98, "Paciencia ($P$ épocas)", ha="center", fontsize=9, fontweight="bold")

    # Flecha de restauración
    ax.annotate(
        "Restaurar pesos θ* para evaluación final",
        xy=(ep_opt, 0.3),
        xytext=(ep_stop, 0.3),
        ha="center",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=2, linestyle=":"),
        fontsize=9,
        fontweight="bold",
        color=TEAL,
    )

    ax.set_title("Mecánica de Early Stopping y Checkpointing Óptimo", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Épocas de Entrenamiento", fontsize=11)
    ax.set_ylabel("Función de Pérdida ($MSE$)", fontsize=11)
    ax.set_xlim(1, 58)
    ax.set_ylim(0, 1.4)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(CARPETA_SALIDA / "esquema_early_stopping.pdf")
    fig.savefig(CARPETA_SALIDA / "esquema_early_stopping.png")
    plt.close(fig)
    print("Figura guardada: esquema_early_stopping.pdf / .png")


def generar_diagrama_residuos_teorico():
    """Genera comparación entre residuos bien calibrados vs residuos sesgados/heterocedásticos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)

    np.random.seed(42)
    n = 250
    y_real = np.linspace(20, 1000, n)

    # Caso 1: Residuos deseables (homocedásticos, centrados en 0)
    res_ideal = np.random.normal(0, 75, n)
    ax1.scatter(y_real, res_ideal, alpha=0.6, color=TEAL, s=30)
    ax1.axhline(0, color=CORAL, linestyle="--", linewidth=1.8)
    ax1.set_title("Residuos Homocedásticos (Bien Calibrado)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Valor Real ($y$)", fontsize=10)
    ax1.set_ylabel("Residuo ($y - \hat{y}$)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.text(
        0.05, 0.90, "✓ Varianza constante\n✓ Sin sesgo sistemático",
        transform=ax1.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="#E8F8F5", edgecolor=TEAL, alpha=0.9)
    )

    # Caso 2: Residuos con sesgo en extremos (típico en demanda con cola pesada)
    res_sesgo = np.random.normal(0, 40, n) + 0.0003 * (y_real - 200) ** 2 - 40
    ax2.scatter(y_real, res_sesgo, alpha=0.6, color=PURPLE, s=30)
    ax2.axhline(0, color=CORAL, linestyle="--", linewidth=1.8)
    ax2.set_title("Residuos con Sesgo Sistemático / Heterocedasticidad", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Valor Real ($y$)", fontsize=10)
    ax2.set_ylabel("Residuo ($y - \hat{y}$)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.55)
    ax2.text(
        0.05, 0.90, "⚠ Subestimación en picos\n⚠ Varianza no constante",
        transform=ax2.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="#FDEDEC", edgecolor=CORAL, alpha=0.9)
    )

    fig.suptitle("Diagnóstico Visual mediante Análisis de Residuos", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(CARPETA_SALIDA / "analisis_residuos_teorico.pdf")
    fig.savefig(CARPETA_SALIDA / "analisis_residuos_teorico.png")
    plt.close(fig)
    print("Figura guardada: analisis_residuos_teorico.pdf / .png")


if __name__ == "__main__":
    generar_esquema_early_stopping()
    generar_diagrama_residuos_teorico()
