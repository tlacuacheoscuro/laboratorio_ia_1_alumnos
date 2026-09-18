#!/usr/bin/env python3
"""Generador de datos sintéticos: edad -> estatura

Genera un CSV con columnas: age,height
"""
import argparse
import numpy as np
import pandas as pd


def generate(n, seed=42):
    rng = np.random.default_rng(seed)
    # edades entre 0 y 18 años
    ages = rng.uniform(0, 18, size=n)
    # curva no lineal aproximada: base + linear + small quadratic
    heights = 45 + 5.2 * ages + 0.08 * (ages ** 2) + rng.normal(0, 3.0, size=n)
    df = pd.DataFrame({"age": np.round(ages, 2), "height": np.round(heights, 2)})
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="Número de muestras")
    p.add_argument("--out", type=str, default="data/estatura_ninos.csv", help="Archivo CSV de salida")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = generate(args.n, seed=args.seed)
    # crear carpeta si no existe
    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generado {len(df)} muestras en {args.out}")


if __name__ == "__main__":
    main()
