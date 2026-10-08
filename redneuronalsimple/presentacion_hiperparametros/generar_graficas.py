#!/usr/bin/env python3
"""
Script para generar las figuras teóricas del espacio de solución y dinámica de optimización
para la presentación en Beamer sobre los parámetros libres de la Red Neuronal.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Crear directorio de salida
OUT_DIR = "graficas"
os.makedirs(OUT_DIR, exist_ok=True)

# Configuración estética general de matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Paleta de colores elegante y moderna
COLOR_TRAIN = '#1f77b4'     # Azul acero
COLOR_TEST  = '#e63946'     # Rojo coral
COLOR_OPT   = '#2a9d8f'     # Verde azulado
COLOR_WARN  = '#f4a261'     # Naranja cálido
COLOR_ACCENT = '#6a0dad'    # Púrpura elegante
COLOR_BG    = '#f8f9fa'     # Gris muy claro


# ==============================================================================
# FIGURA 1: TASA DE APRENDIZAJE (LEARNING RATE)
# ==============================================================================
def generar_figura_learning_rate():
    print("Generando Figura 1: Tasa de Aprendizaje...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Paisaje cuadrático anisotrópico: J(w1, w2) = 0.5 * (w1^2 + 8 * w2^2)
    w1 = np.linspace(-4, 4, 200)
    w2 = np.linspace(-2, 2, 200)
    W1, W2 = np.meshgrid(w1, w2)
    J = 0.5 * (W1**2 + 8 * W2**2)

    # 3 Escenarios de Learning Rate
    tasas = [
        (0.04, r'Baja ($\eta = 0.04$)', '#1f77b4', 'Lenta convergencia\nPasos diminutos'),
        (0.20, r'Óptima ($\eta = 0.20$)', '#2a9d8f', 'Rápida convergencia\nTrayectoria directa'),
        (0.26, r'Excesiva ($\eta = 0.26$)', '#e63946', 'Oscilaciones violentas\n(Inestabilidad/Divergencia)')
    ]

    w_start = np.array([-3.2, 1.6])

    for ax, (lr, titulo, col, desc) in zip(axes, tasas):
        ax.set_facecolor('#ffffff')
        cs = ax.contour(W1, W2, J, levels=14, cmap='Blues_r', alpha=0.6, linewidths=0.9)
        ax.clabel(cs, inline=1, fontsize=7, fmt='%.1f')

        # Simulación de descenso de gradiente
        trajectory = [w_start.copy()]
        curr = w_start.copy()

        # Hessiano: diag([1, 8]) -> gradiente: [w1, 8*w2]
        n_steps = 22 if lr <= 0.20 else 14
        for _ in range(n_steps):
            grad = np.array([curr[0], 8.0 * curr[1]])
            curr = curr - lr * grad
            trajectory.append(curr.copy())
            if np.linalg.norm(curr) > 10:
                break

        traj = np.array(trajectory)

        # Dibujar trayectoria con flechas
        ax.plot(traj[:, 0], traj[:, 1], color=col, marker='o', markersize=4,
                linewidth=1.8, label='Trayectoria', zorder=4)
        for i in range(len(traj)-1):
            dx = traj[i+1, 0] - traj[i, 0]
            dy = traj[i+1, 1] - traj[i, 1]
            ax.annotate('', xy=(traj[i+1, 0], traj[i+1, 1]), xytext=(traj[i, 0], traj[i, 1]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, shrinkA=3, shrinkB=3))

        ax.plot(0, 0, marker='*', markersize=12, color='#e76f51', label='Mínimo global $w^*$', zorder=5)
        ax.plot(w_start[0], w_start[1], marker='s', markersize=7, color='#264653', label='Inicio $w_0$', zorder=5)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-2, 2)
        ax.set_title(f"{titulo}", color=col)
        ax.set_xlabel(r'Parámetro $w_1$')
        if ax == axes[0]:
            ax.set_ylabel(r'Parámetro $w_2$')
        ax.grid(True, linestyle=':', alpha=0.6)
        x_pos = 0.50 if lr > 0.25 else 0.05
        y_pos = 0.08
        ax.text(x_pos, y_pos, desc, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.92, edgecolor='#cccccc'))
        if ax == axes[1]:
            ax.legend(loc='upper right', framealpha=0.9, fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_learning_rate.pdf")
    plt.savefig(f"{OUT_DIR}/fig1_learning_rate.png")
    plt.close()


# ==============================================================================
# FIGURA 2: ESCALADO DE DATOS (STANDARD SCALER)
# ==============================================================================
def generar_figura_escalado():
    print("Generando Figura 2: Escalado de Datos...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # 1. Sin escalar: Hessiano desbalanceado H = diag([0.1, 10.0]) -> Condición kappa = 100
    w1_raw = np.linspace(-3.5, 3.5, 200)
    w2_raw = np.linspace(-1.2, 1.2, 200)
    W1_r, W2_r = np.meshgrid(w1_raw, w2_raw)
    J_raw = 0.5 * (0.2 * W1_r**2 + 8.0 * W2_r**2)

    cs1 = ax1.contour(W1_r, W2_r, J_raw, levels=14, cmap='PuBu_r', alpha=0.7)
    ax1.clabel(cs1, inline=1, fontsize=7, fmt='%.1f')

    # Simulación sin escalar (zig-zag)
    curr = np.array([-3.0, 1.0])
    traj_raw = [curr.copy()]
    lr_raw = 0.22
    for _ in range(16):
        grad = np.array([0.2 * curr[0], 8.0 * curr[1]])
        curr = curr - lr_raw * grad
        traj_raw.append(curr.copy())
    traj_raw = np.array(traj_raw)

    ax1.plot(traj_raw[:, 0], traj_raw[:, 1], color='#d62828', marker='o', markersize=4,
             linewidth=1.8, label=r'Zig-zag ineficiente ($\kappa \gg 1$)')
    ax1.plot(0, 0, marker='*', markersize=12, color='#2a9d8f', label='Óptimo')
    ax1.set_title("SIN Escalado (Variables con distinta escala)\nContornos elípticos muy elongados", color='#d62828')
    ax1.set_xlabel(r'Peso $w_1$ (Checkout Price, escala ~300)')
    ax1.set_ylabel(r'Peso $w_2$ (Promoción, escala 0 a 1)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.text(0.05, 0.08, r"$\nabla J$ es ortogonal a contornos," + "\nno apunta al mínimo. Rebote lateral.",
             transform=ax1.transAxes, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

    # 2. Con StandardScaler: Hessiano isotrópico H = diag([2.0, 2.0]) -> Condición kappa = 1
    w1_s = np.linspace(-3.5, 3.5, 200)
    w2_s = np.linspace(-3.5, 3.5, 200)
    W1_s, W2_s = np.meshgrid(w1_s, w2_s)
    J_s = 0.5 * (2.0 * W1_s**2 + 2.0 * W2_s**2)

    cs2 = ax2.contour(W1_s, W2_s, J_s, levels=12, cmap='PuBu_r', alpha=0.7)
    ax2.clabel(cs2, inline=1, fontsize=7, fmt='%.1f')

    curr = np.array([-3.0, 3.0])
    traj_s = [curr.copy()]
    lr_s = 0.35
    for _ in range(6):
        grad = np.array([2.0 * curr[0], 2.0 * curr[1]])
        curr = curr - lr_s * grad
        traj_s.append(curr.copy())
    traj_s = np.array(traj_s)

    ax2.plot(traj_s[:, 0], traj_s[:, 1], color='#2a9d8f', marker='o', markersize=4,
             linewidth=1.8, label=r'Descenso radial directo ($\kappa \approx 1$)')
    ax2.plot(0, 0, marker='*', markersize=12, color='#2a9d8f', label='Óptimo')
    ax2.set_title("CON StandardScaler (Media 0, Varianza 1)\nContornos esféricos / isotrópicos", color='#2a9d8f')
    ax2.set_xlabel(r'Peso $w_1$ (Estandarizado)')
    ax2.set_ylabel(r'Peso $w_2$ (Estandarizado)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.text(0.05, 0.08, r"$\nabla J$ apunta directamente al óptimo." + "\nConvergencia en mínimos pasos.",
             transform=ax2.transAxes, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_scaling.pdf")
    plt.savefig(f"{OUT_DIR}/fig2_scaling.png")
    plt.close()


# ==============================================================================
# FIGURA 3: TAMAÑO DE LOTE (BATCH SIZE) Y DINÁMICA ESTOCÁSTICA
# ==============================================================================
def generar_figura_batch_size():
    print("Generando Figura 3: Tamaño de Lote...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # 1. Paisaje con dos mínimos: uno agudo (malo para test) y uno plano (bueno para test)
    # Función 1D: potencial doble pozo
    x = np.linspace(-3.5, 3.5, 400)
    # Mínimo agudo en x = -1.8, mínimo plano amplio en x = 1.8
    loss_train = 0.15 * (x**4 - 7*x**2 + 2*x) + 3.0
    loss_test  = loss_train + 0.8 * np.exp(-((x + 1.8)**2)/0.15) - 0.2 * np.exp(-((x - 1.8)**2)/1.2)

    ax1.plot(x, loss_train, color='#1f77b4', linewidth=2.2, label='Pérdida Entrenamiento')
    ax1.plot(x, loss_test, color='#e63946', linewidth=2.0, linestyle='--', label='Pérdida Generalización (Test)')

    # Marcar los dos mínimos
    ax1.annotate('Mínimo Agudo (Sharp)\nBatch Grande (B=N)\nMemoria excesiva, peor test',
                 xy=(-1.9, 0.7), xytext=(-3.3, 3.8),
                 arrowprops=dict(facecolor='#d62828', edgecolor='#d62828', arrowstyle="->", lw=1.5),
                 fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='#d62828'))

    ax1.annotate('Mínimo Plano (Flat)\nMini-Batch (B=64)\nRuido térmico escapa de trampas',
                 xy=(1.85, 0.3), xytext=(0.5, 3.8),
                 arrowprops=dict(facecolor='#2a9d8f', edgecolor='#2a9d8f', arrowstyle="->", lw=1.5),
                 fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='#efe', edgecolor='#2a9d8f'))

    ax1.set_title("Geometría del Espacio: Mínimos Agudos vs Planos", fontsize=11)
    ax1.set_xlabel("Espacio de Parámetros $\\theta$")
    ax1.set_ylabel("Pérdida $J(\\theta)$")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower center', fontsize=8)

    # 2. Trayectorias 2D comparativas según Batch Size
    w1 = np.linspace(-3, 3, 200)
    w2 = np.linspace(-3, 3, 200)
    W1, W2 = np.meshgrid(w1, w2)
    J = W1**2 + W2**2 + 0.5 * np.sin(2.5*W1) * np.cos(2.5*W2)

    cs = ax2.contour(W1, W2, J, levels=14, cmap='Blues_r', alpha=0.6)

    # Trayectoria Full Batch (B = N): suave, determinista
    np.random.seed(42)
    t_full = [[-2.5, 2.5]]
    curr = np.array([-2.5, 2.5])
    for _ in range(12):
        grad = 2.0 * curr
        curr = curr - 0.15 * grad
        t_full.append(curr.copy())
    t_full = np.array(t_full)

    # Trayectoria Mini-Batch (B = 64): balanceada, estocástica moderada
    t_mini = [[-2.5, 2.5]]
    curr = np.array([-2.5, 2.5])
    for _ in range(15):
        grad = 2.0 * curr + np.random.normal(0, 0.6, 2)
        curr = curr - 0.15 * grad
        t_mini.append(curr.copy())
    t_mini = np.array(t_mini)

    # Trayectoria SGD Puro (B = 1): altamente ruidosa
    t_sgd = [[-2.5, 2.5]]
    curr = np.array([-2.5, 2.5])
    for _ in range(18):
        grad = 2.0 * curr + np.random.normal(0, 2.2, 2)
        curr = curr - 0.12 * grad
        t_sgd.append(curr.copy())
    t_sgd = np.array(t_sgd)

    ax2.plot(t_full[:, 0], t_full[:, 1], color='#e76f51', marker='^', markersize=4,
             linewidth=2.0, label='Full Batch ($B=N$): Determinista')
    ax2.plot(t_mini[:, 0], t_mini[:, 1], color='#2a9d8f', marker='o', markersize=3.5,
             linewidth=1.8, label='Mini-Batch ($B=64$): Óptimo')
    ax2.plot(t_sgd[:, 0], t_sgd[:, 1], color='#8338ec', marker='.', markersize=4,
             linewidth=1.2, linestyle=':', alpha=0.8, label='Puro Estocástico ($B=1$): Ruido alto')
    ax2.plot(0, 0, marker='*', markersize=12, color='#e63946', label='Centro Óptimo')

    ax2.set_title("Navegación en el Espacio según Batch Size", fontsize=11)
    ax2.set_xlabel(r'Parámetro $w_1$')
    ax2.set_ylabel(r'Parámetro $w_2$')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_batch_size.pdf")
    plt.savefig(f"{OUT_DIR}/fig3_batch_size.png")
    plt.close()


# ==============================================================================
# FIGURA 4: NÚMERO DE ÉPOCAS Y SOBREAJUSTE (OVERFITTING)
# ==============================================================================
def generar_figura_epocas():
    print("Generando Figura 4: Épocas y Sobreajuste...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # 1. Curvas de aprendizaje teóricas (Train vs Test Loss)
    epochs = np.linspace(1, 150, 300)

    train_loss = 800 * np.exp(-epochs / 25) + 120 * np.exp(-epochs / 70) + 40
    test_loss  = 800 * np.exp(-epochs / 25) + 120 * np.exp(-epochs / 70) + 40 + 0.035 * (epochs - 65)**2 * (epochs > 65)

    ax1.plot(epochs, train_loss, color='#1f77b4', linewidth=2.2, label='Pérdida de Entrenamiento ($J_{train}$)')
    ax1.plot(epochs, test_loss, color='#e63946', linewidth=2.2, label='Pérdida de Prueba ($J_{test}$)')

    # Zona de Early Stopping
    opt_epoch = 65
    ax1.axvline(opt_epoch, color='#2a9d8f', linestyle='--', linewidth=1.8, label=f'Parada Temprana (~{opt_epoch} épocas)')

    # Regiones sombreadas con anotaciones limpias
    ax1.axvspan(1, 30, alpha=0.15, color='#457b9d')
    ax1.axvspan(30, 90, alpha=0.12, color='#2a9d8f')
    ax1.axvspan(90, 150, alpha=0.15, color='#e63946')
    ax1.text(15, 540, "Subajuste", ha='center', fontsize=7.5, color='#1d3557', fontweight='bold')
    ax1.text(60, 540, "Zona Óptima", ha='center', fontsize=7.5, color='#2a9d8f', fontweight='bold')
    ax1.text(120, 540, "Sobreajuste", ha='center', fontsize=7.5, color='#e63946', fontweight='bold')

    ax1.set_title("Dinámica Temporal de Pérdida vs Épocas", fontsize=11)
    ax1.set_xlabel("Número de Épocas")
    ax1.set_ylabel("Pérdida MSE")
    ax1.set_ylim(0, 600)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)

    # 2. Espacio de hipótesis f(x): Underfitting vs Óptimo vs Overfitting
    np.random.seed(42)
    x_pts = np.linspace(-2, 2, 18)
    y_true_fn = lambda x: 0.8 * x**3 - 1.2 * x + 0.2
    y_pts = y_true_fn(x_pts) + np.random.normal(0, 0.45, len(x_pts))

    x_dense = np.linspace(-2.2, 2.2, 200)

    # Modelos
    y_under = 0.5 * x_dense                      # Lineal simple (pocas épocas)
    y_good  = 0.78 * x_dense**3 - 1.15 * x_dense # Óptimo
    poly_over = np.poly1d(np.polyfit(x_pts, y_pts, 12)) # Sobreajustado
    y_over  = poly_over(x_dense)

    ax2.scatter(x_pts, y_pts, color='#333333', s=35, zorder=5, label='Datos reales con ruido')
    ax2.plot(x_dense, y_under, color='#457b9d', linestyle=':', linewidth=2, label='Pocas épocas (Subajuste)')
    ax2.plot(x_dense, y_good, color='#2a9d8f', linewidth=2.4, label='Épocas óptimas (~100)')
    ax2.plot(x_dense, y_over, color='#e63946', linestyle='--', linewidth=1.8, label='Épocas excesivas (Memorización)')

    ax2.set_ylim(-3.5, 3.5)
    ax2.set_title("Efecto en el Espacio de Hipótesis $\\mathcal{H}$", fontsize=11)
    ax2.set_xlabel("Característica de Entrada $x$ (ej. Precio)")
    ax2.set_ylabel("Predicción $y$ (ej. Pedidos)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_epochs_overfitting.pdf")
    plt.savefig(f"{OUT_DIR}/fig4_epochs_overfitting.png")
    plt.close()


# ==============================================================================
# FIGURA 5: ARQUITECTURA: CAPACIDAD Y PROFUNDIDAD
# ==============================================================================
def generar_figura_arquitectura():
    print("Generando Figura 5: Arquitectura (Capas y Neuronas)...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    np.random.seed(10)
    # Generar datos sintéticos de demanda: interacción no lineal precio x promoción
    n_samples = 40
    p = np.random.uniform(-1.5, 1.5, n_samples)
    y_demand = 2.0 / (1.0 + np.exp(2.5 * p)) + np.random.normal(0, 0.15, n_samples)

    p_grid = np.linspace(-1.8, 1.8, 200)

    # 1. 0 capas ocultas (Regresión Lineal Simple): 5 -> 1
    ax1 = axes[0]
    ax1.scatter(p, y_demand, color='#555555', s=25, alpha=0.7, label='Muestras')
    ax1.plot(p_grid, -0.6 * p_grid + 1.0, color='#e63946', linewidth=2.2, label='Ajuste Lineal')
    ax1.set_title("0 Capas Ocultas (Lineal)\nCapacidad Insuficiente", color='#e63946')
    ax1.set_xlabel("Precio (normalizado)")
    ax1.set_ylabel("Demanda")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=7.5)
    ax1.text(0.05, 0.08, "No captura saturación\nni no-linealidad", transform=ax1.transAxes, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='#e63946'))

    # 2. 2 capas ocultas (16 -> 8 -> 1 con ReLU): El modelo de red_neuronal.py
    ax2 = axes[1]
    ax2.scatter(p, y_demand, color='#555555', s=25, alpha=0.7, label='Muestras')
    # Función sigmoide / tanh aproximada por piezas lineales de ReLU
    y_relu = 2.0 / (1.0 + np.exp(2.5 * p_grid))
    ax2.plot(p_grid, y_relu, color='#2a9d8f', linewidth=2.4, label='Red 16 $\\to$ 8 (ReLU)')
    ax2.set_title("2 Capas Ocultas [16, 8]\nBalance Sesgo-Varianza Óptimo", color='#2a9d8f')
    ax2.set_xlabel("Precio (normalizado)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=7.5)
    ax2.text(0.05, 0.08, "Aproxima suavemente curvas;\nEmbudo previene memorización", transform=ax2.transAxes, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#efe', edgecolor='#2a9d8f'))

    # 3. Sobre-parametrizada (ej. 4 capas de 256 neuronas sin regularizar)
    ax3 = axes[2]
    ax3.scatter(p, y_demand, color='#555555', s=25, alpha=0.7, label='Muestras')
    # Curva con oscilaciones intermedias memorizando puntos
    p_sort = np.sort(p)
    from scipy.interpolate import make_interp_spline
    p_sub = p_sort[::3]
    y_sub = y_demand[np.argsort(p)][::3]
    spl = make_interp_spline(p_sub, y_sub, k=3)
    y_over = spl(np.clip(p_grid, p_sub[0], p_sub[-1]))
    ax3.plot(p_grid, y_over, color='#6a0dad', linewidth=2.0, label='Red Sobredimensionada')
    ax3.set_title("Red Gigante (>512 neuronas)\nSobreajuste y Picos Espurios", color='#6a0dad')
    ax3.set_xlabel("Precio (normalizado)")
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(fontsize=7.5)
    ax3.text(0.05, 0.08, "Memoriza ruido muestral;\nPredicciones aberrantes fuera de muestra", transform=ax3.transAxes, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e8fd', edgecolor='#6a0dad'))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_architecture.pdf")
    plt.savefig(f"{OUT_DIR}/fig5_architecture.png")
    plt.close()


# ==============================================================================
# FIGURA 6: FUNCIONES DE ACTIVACIÓN (RELU vs LINEAL vs SIGMOIDE)
# ==============================================================================
def generar_figura_activaciones():
    print("Generando Figura 6: Funciones de Activación...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    z = np.linspace(-4, 4, 300)

    # 1. Funciones de activación
    relu = np.maximum(0, z)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    linear = z / 4.0  # escalada para visualización

    ax1.plot(z, relu, color='#2a9d8f', linewidth=2.4, label=r'ReLU: $\max(0, z)$')
    ax1.plot(z, sigmoid, color='#1f77b4', linewidth=2.0, label=r'Sigmoide: $\frac{1}{1 + e^{-z}}$')
    ax1.plot(z, linear, color='#e63946', linestyle='--', linewidth=1.8, label=r'Lineal (Sin activación): $z$')

    ax1.set_ylim(-0.5, 3.5)
    ax1.set_title("Respuestas de Activación $\\sigma(z)$", fontsize=11)
    ax1.set_xlabel("Entrada Neta $z = w^T x + b$")
    ax1.set_ylabel("Salida Activada $a$")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=8)

    # 2. Gradientes / Derivadas
    d_relu = np.where(z > 0, 1.0, 0.0)
    d_sigmoid = sigmoid * (1.0 - sigmoid)
    d_linear = np.ones_like(z) * 0.25

    ax2.plot(z, d_relu, color='#2a9d8f', linewidth=2.4, label=r"Derivada ReLU: $\sigma'(z) \in \{0, 1\}$")
    ax2.plot(z, d_sigmoid, color='#1f77b4', linewidth=2.0, label=r"Derivada Sigmoide: $\leq 0.25$ (Saturación)")
    ax2.plot(z, d_linear, color='#e63946', linestyle='--', linewidth=1.8, label=r"Derivada Lineal: Constante")

    # Anotaciones
    ax2.annotate('Gradiente activo = 1\n(Sin desvanecimiento)', xy=(2, 1.0), xytext=(0.8, 0.72),
                 arrowprops=dict(arrowstyle="->", color='#2a9d8f', lw=1.3),
                 fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='#efe', edgecolor='#2a9d8f'))

    ax2.annotate('Zona Muerta (Dying ReLU)\nGradiente = 0', xy=(-2.5, 0.0), xytext=(-3.8, 0.40),
                 arrowprops=dict(arrowstyle="->", color='#d62828', lw=1.3),
                 fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='#d62828'))

    ax2.set_ylim(-0.1, 1.3)
    ax2.set_title(r"Derivada del Gradiente $\sigma'(z)$ (Propagación)", fontsize=11)
    ax2.set_xlabel("Entrada Neta $z$")
    ax2.set_ylabel("Magnitud del Gradiente Local")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=7.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig6_activations.pdf")
    plt.savefig(f"{OUT_DIR}/fig6_activations.png")
    plt.close()


# ==============================================================================
# FIGURA 7: OPTIMIZADORES (ADAM vs SGD vs SGD+MOMENTUM)
# ==============================================================================
def generar_figura_optimizadores():
    print("Generando Figura 7: Optimizadores en Valle Estrecho...")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    # Superficie con valle curvado no convexo
    w1 = np.linspace(-3.2, 3.2, 250)
    w2 = np.linspace(-1.2, 3.0, 250)
    W1, W2 = np.meshgrid(w1, w2)
    J = 0.5 * (0.2 * W1**2 + 5.0 * (W2 - 0.25 * W1**2)**2)

    cs = ax.contour(W1, W2, J, levels=np.logspace(-1.0, 1.8, 18), cmap='cividis', alpha=0.6, linewidths=0.9)
    ax.clabel(cs, inline=1, fontsize=6.5, fmt='%.1f')

    start_point = np.array([-2.4, 2.2])

    def grad_fn(w):
        g1 = 0.2 * w[0] - 2.5 * w[0] * (w[1] - 0.25 * w[0]**2)
        g2 = 5.0 * (w[1] - 0.25 * w[0]**2)
        return np.array([g1, g2])

    # 1. SGD estándar
    traj_sgd = [start_point.copy()]
    curr = start_point.copy()
    lr_s = 0.15
    for _ in range(50):
        g = grad_fn(curr)
        curr = curr - lr_s * np.clip(g, -8, 8)
        traj_sgd.append(curr.copy())
    traj_sgd = np.array(traj_sgd)

    # 2. SGD con Momentum
    traj_mom = [start_point.copy()]
    curr = start_point.copy()
    v = np.zeros(2)
    beta = 0.82
    lr_m = 0.06
    for _ in range(50):
        g = grad_fn(curr)
        v = beta * v + (1 - beta) * g
        curr = curr - lr_m * v
        traj_mom.append(curr.copy())
    traj_mom = np.array(traj_mom)

    # 3. Adam
    traj_adam = [start_point.copy()]
    curr = start_point.copy()
    m = np.zeros(2)
    v_a = np.zeros(2)
    b1, b2, eps = 0.88, 0.999, 1e-8
    lr_a = 0.14
    for t in range(1, 60):
        g = grad_fn(curr)
        m = b1 * m + (1 - b1) * g
        v_a = b2 * v_a + (1 - b2) * (g**2)
        m_hat = m / (1 - b1**t)
        v_hat = v_a / (1 - b2**t)
        curr = curr - lr_a * m_hat / (np.sqrt(v_hat) + eps)
        traj_adam.append(curr.copy())
    traj_adam = np.array(traj_adam)

    ax.plot(traj_sgd[:, 0], traj_sgd[:, 1], color='#e63946', marker='x', markersize=3.5,
            linewidth=1.4, alpha=0.85, label='SGD estándar (rebote lateral)')
    ax.plot(traj_mom[:, 0], traj_mom[:, 1], color='#f4a261', marker='s', markersize=3.5,
            linewidth=1.8, label='SGD + Momentum (inercia)')
    ax.plot(traj_adam[:, 0], traj_adam[:, 1], color='#2a9d8f', marker='o', markersize=4,
            linewidth=2.2, label='Adam (adaptativo, óptimo)')

    ax.plot(0, 0, marker='*', markersize=14, color='#e76f51', label='Mínimo Óptimo $(0,0)$', zorder=6)
    ax.plot(start_point[0], start_point[1], marker='o', markersize=8, color='#1d3557', label='Inicio', zorder=6)

    ax.set_title("Trayectorias en Valle No Convexo", fontsize=11)
    ax.set_xlabel(r'Peso $w_1$ (Eje plano)')
    ax.set_ylabel(r'Peso $w_2$ (Eje escarpado)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower center', fontsize=7.5, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig7_optimizers.pdf")
    plt.savefig(f"{OUT_DIR}/fig7_optimizers.png")
    plt.close()


# ==============================================================================
# FIGURA 8: FUNCIÓN DE PÉRDIDA (MSE vs MAE vs HUBER)
# ==============================================================================
def generar_figura_perdidas():
    print("Generando Figura 8: Funciones de Pérdida...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    err = np.linspace(-3.5, 3.5, 300)

    # 1. Magnitud del Castigo
    mse = err**2
    mae = np.abs(err)
    delta = 1.0
    huber = np.where(np.abs(err) <= delta, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta))

    ax1.plot(err, mse, color='#e63946', linewidth=2.2, label=r'MSE ($\mathcal{L}_2$): Penalización $(y - \hat{y})^2$')
    ax1.plot(err, mae, color='#1f77b4', linewidth=2.0, label=r'MAE ($\mathcal{L}_1$): Penalización $|y - \hat{y}|$')
    ax1.plot(err, huber, color='#2a9d8f', linestyle='--', linewidth=2.0, label=r'Huber ($\delta=1$): Mixto cuadrático/lineal')

    ax1.set_ylim(-0.2, 10)
    ax1.set_title("Penalización del Error Residual $(y - \\hat{y})$", fontsize=11)
    ax1.set_xlabel("Error Residual $e = y - \\hat{y}$")
    ax1.set_ylabel("Valor de Pérdida $\\mathcal{L}(e)$")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper center', fontsize=8)
    ax1.text(0.05, 0.72, "MSE penaliza drásticamente\noutliers (errores grandes)", transform=ax1.transAxes, fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='#e63946'))

    # 2. Gradiente del Error
    grad_mse = 2.0 * err
    grad_mae = np.sign(err)
    grad_huber = np.where(np.abs(err) <= delta, err, delta * np.sign(err))

    ax2.plot(err, grad_mse, color='#e63946', linewidth=2.2, label=r"Gradiente MSE: $\frac{\partial \mathcal{L}}{\partial e} = 2e$ (Explosión con outliers)")
    ax2.plot(err, grad_mae, color='#1f77b4', linewidth=2.0, label=r"Gradiente MAE: $\text{sign}(e)$ (Paso constante, salto en 0)")
    ax2.plot(err, grad_huber, color='#2a9d8f', linestyle='--', linewidth=2.0, label=r"Gradiente Huber: Proporcional y luego saturado")

    ax2.set_ylim(-7, 7)
    ax2.set_title(r"Gradiente Local $\partial \mathcal{L} / \partial e$ (Fuerza de Actualización)", fontsize=11)
    ax2.set_xlabel("Error Residual $e = y - \\hat{y}$")
    ax2.set_ylabel("Magnitud del Gradiente")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8_loss_functions.pdf")
    plt.savefig(f"{OUT_DIR}/fig8_loss_functions.png")
    plt.close()


if __name__ == '__main__':
    generar_figura_learning_rate()
    generar_figura_escalado()
    generar_figura_batch_size()
    generar_figura_epocas()
    generar_figura_arquitectura()
    generar_figura_activaciones()
    generar_figura_optimizadores()
    generar_figura_perdidas()
    print("¡Todas las figuras han sido generadas exitosamente en el directorio 'graficas/'!")
