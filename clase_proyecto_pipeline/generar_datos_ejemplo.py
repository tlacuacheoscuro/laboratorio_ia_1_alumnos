"""Genera un CSV pequeno y reproducible para ejemplificar el pipeline completo."""

import csv
from datetime import date, timedelta
from pathlib import Path
import random


ARCHIVO = Path(__file__).with_name("datos_ejemplo_pipeline.csv")


def main():
    generador = random.Random(20261110)
    inicio = date(2026, 3, 1)
    sectores = ["Norte", "Centro", "Sur"]
    filas = []

    for dia in range(180):
        fecha = inicio + timedelta(days=dia)
        sector = sectores[dia % len(sectores)]
        temperatura_c = round(generador.uniform(12.0, 32.0), 1)
        equipos_activos = generador.randint(8, 35)
        mantenimiento = 1 if (dia % 19 == 0) else 0
        ruido = generador.gauss(0.0, 4.2)

        # Relacion sintetica: mas equipos y mayor temperatura elevan el consumo;
        # en dias de mantenimiento preventivo hay menor carga operativa.
        consumo_kwh = round(
            25.0
            + 3.1 * equipos_activos
            + 1.35 * temperatura_c
            - 14.0 * mantenimiento
            + ruido,
            1,
        )

        filas.append({
            "fecha": fecha.isoformat(),
            "sector": sector,
            "temperatura_c": temperatura_c,
            "equipos_activos": equipos_activos,
            "mantenimiento": mantenimiento,
            "consumo_kwh": max(15.0, consumo_kwh),
        })

    # Casos deliberados para que la auditoria del pipeline tenga trabajo real:
    # 1) Valor faltante
    filas[28]["temperatura_c"] = ""
    # 2) Error de captura evidente (valor negativo imposible)
    filas[64]["equipos_activos"] = -5
    # 3) Dos picos extremos reales o anomalos para investigar con IQR
    filas[95]["consumo_kwh"] = 215.0
    filas[142]["consumo_kwh"] = 230.0

    campos = [
        "fecha",
        "sector",
        "temperatura_c",
        "equipos_activos",
        "mantenimiento",
        "consumo_kwh",
    ]
    with ARCHIVO.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"Archivo creado: {ARCHIVO.name} ({len(filas)} filas)")


if __name__ == "__main__":
    main()

