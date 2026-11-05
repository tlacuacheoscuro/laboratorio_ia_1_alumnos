"""Analiza un CSV y predice unidades con regresion lineal basica."""

import csv
from pathlib import Path

import numpy as np


ARCHIVO = Path(__file__).with_name("ventas_clase.csv")
ENTRADAS = ["visitas", "descuento_pct"]
OBJETIVO = "unidades"


def main():
    with ARCHIVO.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        datos = list(lector)
        columnas = lector.fieldnames

    print("Columnas:", columnas)
    print("Primeras filas:")
    for fila in datos[:5]:
        print(fila)
    print("\nValores faltantes:")
    for columna in columnas:
        faltantes = sum(fila[columna] == "" for fila in datos)
        print(f"{columna}: {faltantes}")

    limpios = []
    for fila in datos:
        try:
            for columna in ENTRADAS + [OBJETIVO, "precio_unitario"]:
                fila[columna] = float(fila[columna])
        except ValueError:
            continue
        fila["importe"] = fila[OBJETIVO] * fila["precio_unitario"]
        limpios.append(fila)

    por_ciudad = {}
    for fila in limpios:
        ciudad = fila["ciudad"]
        por_ciudad[ciudad] = por_ciudad.get(ciudad, 0) + fila["importe"]
    print("\nImporte total por ciudad:")
    for ciudad, importe in sorted(por_ciudad.items(), key=lambda par: -par[1]):
        print(f"{ciudad}: {importe:.2f}")

    ventas = np.array([fila[OBJETIVO] for fila in limpios])
    q1, q3 = np.quantile(ventas, [0.25, 0.75])
    rango = q3 - q1
    atipicos = [
        fila for fila in limpios
        if fila[OBJETIVO] < q1 - 1.5 * rango
        or fila[OBJETIVO] > q3 + 1.5 * rango
    ]
    print("\nPosibles valores atipicos:")
    for fila in atipicos:
        print(fila["fecha"], int(fila[OBJETIVO]))

    corte = int(0.8 * len(limpios))
    entrenamiento = limpios[:corte]
    prueba = limpios[corte:]
    x_train = np.array([[fila[col] for col in ENTRADAS] for fila in entrenamiento])
    x_test = np.array([[fila[col] for col in ENTRADAS] for fila in prueba])
    y_train = np.array([fila[OBJETIVO] for fila in entrenamiento])
    y_test = np.array([fila[OBJETIVO] for fila in prueba])

    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    coeficientes = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    predicciones = x_test @ coeficientes

    error_modelo = np.mean(np.abs(y_test - predicciones))
    prediccion_base = np.full(len(y_test), np.mean(y_train))
    error_base = np.mean(np.abs(y_test - prediccion_base))
    print(f"\nFilas: {len(datos)}; utilizables: {len(limpios)}")
    print(f"MAE de la media (referencia): {error_base:.2f} unidades")
    print(f"MAE del modelo: {error_modelo:.2f} unidades")
    print("Coeficientes [intercepto, visitas, descuento]:", coeficientes.round(3))


if __name__ == "__main__":
    main()