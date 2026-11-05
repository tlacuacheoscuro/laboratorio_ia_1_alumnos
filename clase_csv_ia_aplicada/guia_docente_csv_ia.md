# Plan de clase: Del CSV al modelo

**Duración:** 75 minutos  
**Tema:** Análisis tabular, detección de valores atípicos y regresión supervisada  
**Prerrequisitos:** Python básico, CSV y conceptos iniciales de entrenamiento/evaluación  
**Dependencia externa:** NumPy. La lectura CSV usa la biblioteca estándar de Python.

## Objetivos

Al terminar, el alumnado podrá:

- inspeccionar columnas y valores faltantes de un CSV y justificar una decisión de limpieza;
- crear una columna derivada y resumir datos por grupo;
- explicar por qué un valor atípico se investiga antes de eliminarlo;
- separar entrenamiento y prueba respetando el orden temporal;
- comparar una regresión lineal con una predicción de referencia usando MAE.

## Conexión con lo aprendido

La primera práctica CSV aporta carga, limpieza, filtros y agrupaciones. La clase de redes neuronales aporta el ciclo supervisado de entradas, objetivo, entrenamiento y evaluación. La sesión de anomalías aporta la cautela de investigar datos inusuales sin convertir una señal en una acusación. Los registros del robot ofrecen un conjunto realista para transferir después el mismo proceso.

No se intenta meter todos los modelos vistos en una sola sesión: embeddings/RAG trabajan principalmente con representaciones y búsqueda de texto, mientras que aquí la pregunta es tabular y predictiva.

## Secuencia de 75 minutos

| Minutos | Bloque | Actividad |
|---|---|---|
| 0–8 | Pregunta y datos | ¿Podemos predecir unidades a partir de visitas y descuentos? Revisar las columnas y unidades. |
| 8–20 | Auditoría CSV | Ejecutar el generador; inspeccionar muestra, tipos y faltantes. Discutir la fila incompleta. |
| 20–30 | Transformación | Calcular importe y comparar sumas por ciudad. Aclarar descripción frente a causalidad. |
| 30–40 | Atípicos | Explicar IQR, revisar las fechas detectadas y debatir explicaciones posibles. |
| 40–58 | Modelo supervisado | Formar entrenamiento y prueba en orden temporal; ajustar la regresión y explicar sus coeficientes. |
| 58–68 | Evaluación | Comparar MAE con la referencia de la media; cambiar las variables de entrada y repetir. |
| 68–75 | Transferencia | Diseñar columnas para un CSV del robot y escoger un objetivo que se pueda predecir. |

## Práctica guiada

Desde la carpeta de la clase:

```bash
python3 generar_ventas.py
python3 analizar_ventas.py
```

El generador crea 240 días deterministas, con visitas, descuento, unidades, precio y ciudad. Incluye una celda faltante y dos valores altos deliberados para que las etapas de auditoría sean visibles. `analizar_ventas.py` lee el archivo con `csv.DictReader`, convierte los campos numéricos, resume por ciudad, usa IQR para marcar posibles atípicos, ajusta coeficientes por mínimos cuadrados y evalúa los últimos días.

## Tarea modificable

1. Cambiar `ENTRADAS` a `["visitas"]` y anotar antes de ejecutar si esperan que el error aumente o disminuya.
2. Cambiar el corte de 80/20 a 70/30 y comparar la evaluación.
3. Escoger un valor atípico y explicar al menos dos causas posibles antes de proponer excluirlo.
4. Proponer un esquema de CSV del robot con columnas como `episodio`, `paso`, `choque`, `accion`, `distancia_meta` y `recompensa`; indicar qué columna sería el objetivo.

**Entrega:** cambio realizado, predicción previa, MAE observado y una limitación. No se califica que el modelo dé el mejor número, sino que la comparación sea correcta y razonada.

## Precisión técnica para el docente

- El generador produce una relación aproximadamente lineal con ruido; el conjunto es sintético para que la evaluación sea reproducible.
- La división temporal evita entrenar con días posteriores a los evaluados. No implica que el ejemplo sea una predicción comercial fiable.
- La regresión incluye intercepto y usa `numpy.linalg.lstsq`; es intencionalmente pequeña y no es una red neuronal.
- La media del entrenamiento constituye una referencia simple. El MAE se informa en unidades vendidas.
- El IQR marca extremos estadísticos, no errores confirmados. No se eliminan automáticamente del ajuste.
- Los coeficientes describen relaciones del conjunto generado; no permiten afirmar causalidad.

## Preguntas de cierre

1. ¿Por qué no evaluar el modelo con las mismas filas que usó para ajustarse?
2. ¿Qué problema habría si incluyéramos `importe` como entrada al predecir unidades si importe se calculó directamente usando unidades?
3. ¿Por qué un error promedio puede ocultar un error muy grande en un día concreto?
4. ¿Qué columnas del robot serían sensores disponibles antes de tomar una acción y cuáles solo se conocen después?