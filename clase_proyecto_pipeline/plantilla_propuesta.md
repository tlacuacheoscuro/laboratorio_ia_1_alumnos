# Acta de Inicio del Proyecto Final: Pipeline Completo de IA

**Fecha de arranque:** 10 de noviembre  
**Fecha límite de entrega final:** 15 de enero (~2 meses)  
**Integrantes del equipo:**  
**Título tentativo del proyecto:**  

---

## 1. Ruta elegida (marcar una)

- [ ] **Ruta A — Datos tabulares y predicción supervisada:** CSV propio o de simulación (ej. ventas, energía, demanda, registros del robot), auditoría, regresión o red neuronal, comparación contra referencia.
- [ ] **Ruta B — Representación, espacio latente y patrones:** SOM, Autoencoder o VAE sobre datos numéricos/sensoriales para agrupar, comprimir, visualizar estructura o detectar anomalías.
- [ ] **Ruta C — Texto, embeddings y recuperación (RAG):** Corpus de documentos propios, estrategia de fragmentación (*chunking* con solapamiento), embeddings, búsqueda por similitud coseno y evaluación de recuperación.

---

## 2. La pregunta del proyecto

- **¿Qué pregunta concreta queremos responder o qué tarea queremos resolver?**  
  *(Evitar frases vagas como «usar IA para el tráfico»; escribir algo medible como «estimar el tiempo de recorrido a partir de hora, lluvia y flujo previo»).*
  > Respuesta:

- **¿Qué decisión o análisis apoya este sistema y qué NO puede afirmar?**  
  *(Recuerda: predicción o asociación no equivale a causalidad).*
  > Respuesta:

---

## 3. Datos y Auditoría (Etapas 1, 2 y 3 del Pipeline)

- **Origen de los datos:** *(¿De dónde salen? ¿Cuántas filas/documentos tienen?)*  
  > Respuesta:
- **Variables de entrada ($X$) disponibles antes de predecir:**  
  > Respuesta:
- **Variable objetivo ($y$) o criterio de calidad:**  
  > Respuesta:
- **Riesgo de fuga de información (*data leakage*):** *(¿Qué columna o dato NO debemos incluir en $X$ porque contiene la respuesta o información del futuro?)*  
  > Respuesta:
- **Reglas de limpieza y manejo de atípicos:** *(¿Qué valores son físicamente imposibles y cuáles se marcarán con IQR/Benford para investigar sin borrarlos a ciegas?)*  
  > Respuesta:

---

## 4. Partición, Referencia (*Baseline*) y Modelo (Etapas 4 y 5)

- **Estrategia de partición (Entrenamiento / Validación / Prueba):**  
  *(¿Corte temporal o aleatorio? ¿En qué proporción?)*  
  > Respuesta:
- **Referencia simple (*Baseline*) a vencer:**  
  *(Ej. predecir la media del entrenamiento, repetir el último valor, o búsqueda por palabras exactas en RAG).*  
  > Respuesta:
- **Modelo(s) del curso que implementaremos:**  
  > Respuesta:
- **Métrica principal de evaluación:** *(Ej. MAE, MSE, Error de Cuantización/Topológico en SOM, o Precisión Top-$k$ en RAG).*  
  > Respuesta:

---

## 5. Cronograma de Hitos (10 de noviembre – 15 de enero)

| Hito | Fecha sugerida | Entregable verificable | Estado |
|---|---|---|---|
| **Hito 1: Banderazo y Datos** | Semana 1–2 (noviembre) | Esta acta completa + dataset cargado y auditado (`Etapas 1 y 2`). | [ ] |
| **Hito 2: Referencia y Limpieza** | Semana 3–4 (fines de nov / inicios de dic) | Reporte de atípicos + partición sin fuga + métrica del *Baseline* funcionando (`Etapas 3 y 4`). | [ ] |
| **Hito 3: Modelo e Iteración** | Semana 5–6 (diciembre) | Modelo entrenado superando o compitiendo con la referencia + ajuste de hiperparámetros (`Etapa 5`). | [ ] |
| **Hito 4: Autopsia y Cierre** | Semana 7–8 (hasta el 15 de enero) | Análisis de los peores errores en prueba, gráficas, limitaciones y presentación final (`Etapa 6`). | [ ] |

