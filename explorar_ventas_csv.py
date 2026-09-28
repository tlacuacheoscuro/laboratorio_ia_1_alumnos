from pathlib import Path

import pandas as pd


def main():
    ruta_csv = Path(__file__).with_name("ventas_sinteticas.csv")
    datos = pd.read_csv(ruta_csv, parse_dates=["fecha"])

    print("Muestra de datos:")
    print(datos.head(), end="\n\n")

    print(f"Dimensiones (filas, columnas): {datos.shape}")
    print("\nTipos de columnas:")
    print(datos.dtypes)

    print("\nValores faltantes por columna:")
    print(datos.isna().sum())

    print("\nResumen numerico:")
    columnas_numericas = datos.select_dtypes(include="number")
    print(columnas_numericas.describe())

    datos_limpios = datos.dropna(
        subset=["unidades", "precio_unitario"]
    ).copy()
    datos_limpios["importe"] = (
        datos_limpios["unidades"] * datos_limpios["precio_unitario"]
    )

    print(f"\nFilas originales: {len(datos)}")
    print(f"Filas utilizables: {len(datos_limpios)}")

    ventas_grandes = datos_limpios[datos_limpios["importe"] >= 10]
    print("\nVentas con importe de 10 o mas:")
    print(ventas_grandes.sort_values("importe", ascending=False).to_string(index=False))

    por_categoria = datos_limpios.groupby("categoria").agg(
        ventas_totales=("importe", "sum"),
        unidades_totales=("unidades", "sum"),
        numero_registros=("producto", "count"),
    )
    por_categoria = por_categoria.sort_values(
        "ventas_totales", ascending=False
    )

    print("\nResumen por categoria:")
    print(por_categoria.to_string())

    por_ciudad = datos_limpios.groupby("ciudad")["importe"].sum()
    print("\nVentas totales por ciudad:")
    print(por_ciudad.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()