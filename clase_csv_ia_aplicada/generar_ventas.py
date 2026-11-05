"""Genera un CSV didactico reproducible de actividad diaria de una tienda."""

import csv
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(23)
    cantidad = 240
    visitas = rng.integers(80, 501, size=cantidad)
    descuento = rng.integers(0, 31, size=cantidad)
    unidades = np.rint(
        0.08 * visitas + 0.45 * descuento + rng.normal(0, 4, cantidad)
    ).clip(1).astype(int)
    unidades[[42, 137]] += [55, 40]
    precio = np.round(rng.uniform(4, 18, size=cantidad), 2)
    ciudades = rng.choice(["Santiago", "Valparaiso", "Concepcion"], cantidad)

    destino = Path(__file__).with_name("ventas_clase.csv")
    with destino.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            "fecha", "ciudad", "visitas", "descuento_pct",
            "unidades", "precio_unitario",
        ])
        for indice in range(cantidad):
            fecha = np.datetime64("2026-01-01") + np.timedelta64(indice, "D")
            descuento_dia = "" if indice == 18 else int(descuento[indice])
            escritor.writerow([
                str(fecha), ciudades[indice], int(visitas[indice]),
                descuento_dia, int(unidades[indice]), precio[indice],
            ])
    print(f"CSV generado: {destino.name} ({cantidad} filas)")


if __name__ == "__main__":
    main()