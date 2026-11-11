"""Plantilla ejecutable de un pipeline completo de IA de principio a fin.

No requiere bibliotecas externas (usa unicamente la biblioteca estandar de Python)
para que cualquier equipo pueda ejecutarlo de inmediato como punto de partida.

Etapas incluidas:
1. Carga y auditoria inicial (columnas, tipos, valores faltantes).
2. Limpieza con reglas explicitas y transformacion de variables.
3. Deteccion de valores atipicos (IQR) para investigacion (sin borrado ciego).
4. Particion temporal entrenamiento / prueba (sin fuga de datos).
5. Ajuste de modelo de referencia (media) y modelo lineal supervisado.
6. Evaluacion en prueba, autopsia de los peores errores y exportacion a CSV.
"""

import csv
from pathlib import Path


ARCHIVO_DATOS = Path(__file__).with_name("datos_ejemplo_pipeline.csv")
ARCHIVO_SALIDA = Path(__file__).with_name("predicciones_prueba.csv")

ENTRADAS = ["temperatura_c", "equipos_activos", "mantenimiento"]
OBJETIVO = "consumo_kwh"
PROPORCION_ENTRENAMIENTO = 0.80


def percentil(valores_ordenados, p):
    """Calcula el percentil p in [0, 1] por interpolacion lineal."""
    pos = (len(valores_ordenados) - 1) * p
    inf = int(pos)
    sup = min(inf + 1, len(valores_ordenados) - 1)
    frac = pos - inf
    return valores_ordenados[inf] * (1.0 - frac) + valores_ordenados[sup] * frac


def resolver_sistema_lineal(a, b):
    """Resuelve A * x = b por eliminacion gaussiana con pivoteo parcial."""
    n = len(b)
    m = [fila[:] + [ float(b[i]) ] for i, fila in enumerate(a)]

    for col in range(n):
        pivote = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivote] = m[pivote], m[col]
        divisor = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= divisor
        for r in range(n):
            if r != col:
                factor = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def ajustar_minimos_cuadrados(x_filas, y_valores):
    """Calcula coeficientes [b0, b1, ...] resolviendo (X^T X) b = X^T y."""
    x_con_uno = [[1.0] + list(fila) for fila in x_filas]
    d = len(x_con_uno[0])
    xtx = [[0.0] * d for _ in range(d)]
    xty = [0.0] * d

    for fila, y in zip(x_con_uno, y_valores):
        for i in range(d):
            xty[i] += fila[i] * y
            for j in range(d):
                xtx[i][j] += fila[i] * fila[j]

    return resolver_sistema_lineal(xtx, xty)


def predecir_lineal(x_filas, coeficientes):
    """Aplica b0 + b1*x1 + ... a cada fila."""
    predicciones = []
    for fila in x_filas:
        total = coeficientes[0]
        for valor, peso in zip(fila, coeficientes[1:]):
            total += valor * peso
        predicciones.append(total)
    return predicciones


def cargar_y_auditar(ruta: Path):
    """Etapa 1: Lee el CSV e informa su estado crudo antes de tocar nada."""
    with ruta.open(encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f)
        filas = list(lector)
        columnas = lector.fieldnames or []

    print("=== ETAPA 1: AUDITORIA DE DATOS CRUDOS ===")
    print(f"Archivo: {ruta.name} | Total de filas: {len(filas)}")
    print("Columnas:", ", ".join(columnas))
    for col in columnas:
        vacios = sum(1 for fila in filas if fila[col].strip() == "")
        print(f"  - {col}: {vacios} celdas vacias")
    return filas


def limpiar_y_transformar(filas_crudas):
    """Etapa 2: Convierte tipos, descarta registros invalidos y crea variables."""
    limpios = []
    descartados_vacio = 0
    descartados_rango = 0

    for fila in filas_crudas:
        try:
            temp = float(fila["temperatura_c"])
            equipos = float(fila["equipos_activos"])
            mant = float(fila["mantenimiento"])
            consumo = float(fila["consumo_kwh"])
        except ValueError:
            descartados_vacio += 1
            continue

        # Regla de dominio explicita: no puede haber equipos negativos ni consumo <= 0
        if equipos < 0 or consumo <= 0:
            descartados_rango += 1
            continue

        registro = {
            "fecha": fila["fecha"],
            "sector": fila["sector"],
            "temperatura_c": temp,
            "equipos_activos": equipos,
            "mantenimiento": mant,
            "consumo_kwh": consumo,
            "consumo_por_equipo": consumo / max(equipos, 1.0),
        }
        limpios.append(registro)

    print("\n=== ETAPA 2: LIMPIEZA Y TRANSFORMACION ===")
    print(f"Filas validas conservadas: {len(limpios)}")
    print(f"Descartadas por valor faltante/no numerico: {descartados_vacio}")
    print(f"Descartadas por regla de dominio (valores imposibles): {descartados_rango}")
    return limpios


def detectar_atipicos_iqr(filas, columna):
    """Etapa 3: Marca extremos estadisticos con IQR para investigarlos."""
    valores = sorted(fila[columna] for fila in filas)
    q1 = percentil(valores, 0.25)
    q3 = percentil(valores, 0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr

    atipicos = [
        fila for fila in filas
        if fila[columna] < limite_inf or fila[columna] > limite_sup
    ]

    print("\n=== ETAPA 3: INVESTIGACION DE ATIPICOS (IQR) ===")
    print(f"Variable: {columna} | Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
    print(f"Umbrales de alerta: [{limite_inf:.1f}, {limite_sup:.1f}]")
    print(f"Casos marcados para revision ({len(atipicos)}):")
    for caso in atipicos:
        print(
            f"  * Fecha {caso['fecha']} ({caso['sector']}): "
            f"{columna}={caso[columna]:.1f} | equipos={int(caso['equipos_activos'])}"
        )
    return atipicos


def dividir_datos(filas, entradas, objetivo, proporcion_train):
    """Etapa 4: Separa pasado (entrenamiento) y futuro (prueba) sin fuga."""
    corte = int(len(filas) * proporcion_train)
    train = filas[:corte]
    test = filas[corte:]

    x_train = [[f[col] for col in entradas] for f in train]
    y_train = [f[objetivo] for f in train]
    x_test = [[f[col] for col in entradas] for f in test]
    y_test = [f[objetivo] for f in test]

    print("\n=== ETAPA 4: PARTICION TEMPORAL SIN FUGA DE DATOS ===")
    print(
        f"Entrenamiento: {len(train)} filas "
        f"({train[0]['fecha']} a {train[-1]['fecha']})"
    )
    print(
        f"Prueba:        {len(test)} filas "
        f"({test[0]['fecha']} a {test[-1]['fecha']})"
    )
    return train, test, x_train, y_train, x_test, y_test


def entrenar_y_evaluar(x_train, y_train, x_test, y_test):
    """Etapa 5: Compara una referencia simple (media) contra el modelo."""
    media_train = sum(y_train) / len(y_train)
    mae_base = sum(abs(y - media_train) for y in y_test) / len(y_test)

    coeficientes = ajustar_minimos_cuadrados(x_train, y_train)
    pred_modelo = predecir_lineal(x_test, coeficientes)
    mae_modelo = sum(abs(y - p) for y, p in zip(y_test, pred_modelo)) / len(y_test)

    mejora_pct = 100.0 * (mae_base - mae_modelo) / mae_base
    coefs_txt = [round(c, 3) for c in coeficientes]

    print("\n=== ETAPA 5: MODELADO Y COMPARACION CONTRA REFERENCIA ===")
    print(f"Entradas utilizadas: {ENTRADAS}")
    print(f"Coeficientes [b0, {', '.join(ENTRADAS)}]: {coefs_txt}")
    print(f"MAE Referencia (media = {media_train:.2f}): {mae_base:.2f}")
    print(f"MAE Modelo en prueba:                   {mae_modelo:.2f}")
    print(f"Reduccion relativa del error:           {mejora_pct:.1f}%")
    return pred_modelo, mae_base, mae_modelo, coeficientes


def analizar_errores_y_exportar(test_filas, y_test, pred_modelo, ruta_salida: Path):
    """Etapa 6: Inspecciona donde se equivoca mas el modelo y guarda resultados."""
    errores_abs = [abs(y - p) for y, p in zip(y_test, pred_modelo)]
    indices_ordenados = sorted(
        range(len(errores_abs)), key=lambda i: errores_abs[i], reverse=True
    )

    print("\n=== ETAPA 6: AUTOPSIA DE ERRORES EN PRUEBA ===")
    print("Los 3 dias de prueba con mayor error absoluto:")
    for idx in indices_ordenados[:3]:
        fila = test_filas[idx]
        real = y_test[idx]
        pred = pred_modelo[idx]
        err = errores_abs[idx]
        print(
            f"  * {fila['fecha']} ({fila['sector']}): "
            f"Real={real:.1f}, Predicho={pred:.1f}, Error={err:.1f}"
        )

    with ruta_salida.open("w", encoding="utf-8", newline="") as f:
        campos = ["fecha", "sector", OBJETIVO, "prediccion", "error_absoluto"]
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for fila, real, pred, err in zip(test_filas, y_test, pred_modelo, errores_abs):
            escritor.writerow({
                "fecha": fila["fecha"],
                "sector": fila["sector"],
                OBJETIVO: f"{real:.2f}",
                "prediccion": f"{pred:.2f}",
                "error_absoluto": f"{err:.2f}",
            })
    print(f"\nPredicciones de prueba exportadas a: {ruta_salida.name}")


def main():
    if not ARCHIVO_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontro {ARCHIVO_DATOS.name}. "
            "Ejecuta primero: python3 generar_datos_ejemplo.py"
        )

    filas_crudas = cargar_y_auditar(ARCHIVO_DATOS)
    filas_limpias = limpiar_y_transformar(filas_crudas)
    detectar_atipicos_iqr(filas_limpias, OBJETIVO)
    _, test_filas, x_train, y_train, x_test, y_test = dividir_datos(
        filas_limpias, ENTRADAS, OBJETIVO, PROPORCION_ENTRENAMIENTO
    )
    pred_modelo, _, _, _ = entrenar_y_evaluar(x_train, y_train, x_test, y_test)
    analizar_errores_y_exportar(test_filas, y_test, pred_modelo, ARCHIVO_SALIDA)


if __name__ == "__main__":
    main()
