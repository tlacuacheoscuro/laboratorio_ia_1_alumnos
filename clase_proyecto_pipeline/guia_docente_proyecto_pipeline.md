# Plan de clase y guía docente: El Pipeline Completo de IA (Banderazo del Proyecto Final)

**Duración:** 75 minutos  
**Tema:** Integración del curso, arquitectura de un pipeline reproducible y arranque del proyecto final  
**Horizonte del proyecto:** 10 de noviembre al 15 de enero (~2 meses / 8–9 semanas efectivas)  
**Dependencia externa:** Ninguna (usa la biblioteca estándar de Python para que corra en cualquier máquina desde el día 1).

---

## 1. Propósito de la sesión

A lo largo del semestre el alumnado trabajó con piezas fundamentales pero separadas:
1. **Agentes y simulaciones:** robot reactivo/evolutivo/con memoria y Juego de la Vida.
2. **Datos tabulares y modelos supervisados:** lectura de CSV, limpieza, detección de anomalías (Benford e IQR), regresión lineal y redes neuronales (hiperparámetros, *early stopping*, autopsia de errores).
3. **Espacios latentes y autoorganización:** SOM de Kohonen, Autoencoders deterministas y Autoencoders Variacionales (VAE).
4. **Texto, representación y contexto:** Embeddings semánticos, bases vectoriales / RAG y el mecanismo de auto-atención en Transformers.

El objetivo de hoy es dar el **banderazo inicial** para que durante los próximos **dos meses (hasta el 15 de enero)** cada equipo construya un **proyecto integrador con el pipeline completo**, evitando el error clásico de los proyectos estudiantiles: dedicar el 95 % del tiempo a buscar un modelo complejo a última hora sin auditar datos, sin *baseline* y con fuga de información.

---

## 2. Secuencia didáctica de 75 minutos

| Minutos | Bloque | Actividad en aula |
|---|---|---|
| **0–10** | El mapa del curso | Conectar las clases previas en tres grandes familias (Tabular/Supervisado, Espacio Latente, Texto/RAG) y aclarar que un buen proyecto elige **una** ruta coherente, no mezcla todo sin sentido. |
| **10–25** | Anatomía del Pipeline | Explicar las 6 etapas: 1) Pregunta y auditoría, 2) Limpieza con reglas de dominio, 3) Investigación de atípicos, 4) Partición sin fuga (*data leakage*), 5) Referencia (*baseline*) vs. modelo, 6) Autopsia de errores. |
| **25–42** | Práctica en vivo | Ejecutar `generar_datos_ejemplo.py` y `pipeline_base.py`. Revisar la salida etapa por etapa en terminal y realizar un cambio controlado (quitar una variable de entrada y medir el impacto en MAE). |
| **42–60** | Taller de diseño | En equipos (o individualmente), abrir `plantilla_propuesta.md` y redactar el **Acta de Inicio del Proyecto**: elegir Ruta A, B o C, definir la pregunta, el origen de los datos, la referencia a vencer y el posible riesgo de fuga. |
| **60–75** | Revisión rápida y calendario | Tres equipos leen en voz alta su pregunta y su *baseline* (2 min por equipo) para recibir retroalimentación inmediata en plenaria. Fijar las fechas de los 4 hitos hasta el **15 de enero**. |

---

## 3. Las tres rutas de proyecto sugeridas

Para evitar bloqueos al elegir tema, proponga estas tres rutas conectadas directamente con el repositorio del curso:

### Ruta A: Datos tabulares y predicción supervisada
- **Punto de partida en el curso:** `clase_csv_ia_aplicada/`, `redneuronalsimple/`, registros exportados de `robot_red_neuronal_memoria.py`.
- **Qué debe incluir el pipeline:** auditoría de faltantes y rangos, análisis de valores extremos con IQR/Benford, división temporal o estratificada sin fuga, comparación de un *baseline* (media o persistencia) contra regresión lineal y/o red neuronal con *early stopping*, y análisis de los 5 casos de prueba con mayor error.

### Ruta B: Representación, espacio latente y detección de patrones
- **Punto de partida en el curso:** `som_rgb/`, `autoencoder_espacio_latente/`, `vae_espacio_latente/`.
- **Qué debe incluir el pipeline:** matriz de datos multivariados (sensores, colores, perfiles de consumo o estados del robot), normalización calculada solo en entrenamiento, proyección a 2D con SOM o Autoencoder/VAE, comparación contra una referencia simple (ej. proyección aleatoria o umbral fijo), y análisis de qué representan las regiones del mapa o los casos con mayor error de reconstrucción.

### Ruta C: Texto, embeddings y recuperación aumentada (RAG)
- **Punto de partida en el curso:** `clase_embeddings_semanticos/`, `clase_rag_bases_vectoriales/`, `clase_transformers_atencion/`.
- **Qué debe incluir el pipeline:** corpus de documentos propios (reglamentos, apuntes, manuales técnicos), limpieza del texto, comparación de al menos dos configuraciones de *chunking* (con y sin solapamiento), vectorización y búsqueda por similitud coseno, comparación contra un *baseline* léxico (coincidencia exacta de palabras) sobre un conjunto fijo de 15–20 preguntas de prueba, y autopsia de las preguntas donde recupera el fragmento equivocado.

---

## 4. Calendario de seguimiento (10 de noviembre – 15 de enero)

Con dos meses de margen (incluyendo el receso de fin de año), conviene dividir el proyecto en **4 entregas cortas y acumulativas** para que ningún equipo llegue en blanco a enero:

1. **Hito 1 — Banderazo y Auditoría de Datos (finales de noviembre):**
   - Entrega de `plantilla_propuesta.md` firmada.
   - Script de carga que imprime cuántas filas/documentos hay, qué columnas existen y cuántos faltantes o valores inválidos aparecen (Etapas 1 y 2).
2. **Hito 2 — Partición sin Fuga y Referencia / *Baseline* (primera mitad de diciembre):**
   - Regla de partición entrenamiento/prueba implementada.
   - Detección de atípicos documentada.
   - Métrica del modelo de referencia (*baseline*) ya calculada en prueba (Etapas 3 y 4).
3. **Hito 3 — Modelo Principal e Iteración (regreso en enero, semana previa a la entrega):**
   - Modelo entrenado y comparado en la misma tabla contra el *baseline* (Etapa 5).
   - Al menos un experimento de ablación o cambio de hiperparámetros (ej. quitar una variable, cambiar tamaño de ventana o número de épocas).
4. **Hito 4 — Autopsia Forense y Presentación Final (15 de enero):**
   - Código reproducible de punta a punta (`pipeline`), gráficas finales, inspección detallada de los peores errores en prueba y exposición de límites (Etapa 6).

---

## 5. Práctica en clase con el esqueleto `pipeline_base.py`

Desde la carpeta `clase_proyecto_pipeline/`:

```bash
python3 generar_datos_ejemplo.py
python3 pipeline_base.py
```

### Experimento guiado en clase (10 minutos)
Pida al grupo que antes de tocar el código respondan:
1. *«En `datos_ejemplo_pipeline.csv` hay un día con `-5` equipos activos y dos días con más de `210 kWh`. ¿Por qué el pipeline descarta el `-5` en la Etapa 2 pero conserva los de `210 kWh` en la Etapa 3?»*  
   **Respuesta esperada:** Porque $-5$ equipos es físicamente imposible (error de captura o sensor roto), mientras que un consumo alto puede ser un pico real que debe investigarse antes de decidir si se excluye.
2. *«Si quitamos `equipos_activos` de la lista `ENTRADAS` en `pipeline_base.py`, ¿el MAE en prueba subirá un poco o se acercará al MAE de la media (22.42)?»*  
   Haga que editen `ENTRADAS = ["temperatura_c", "mantenimiento"]`, ejecuten de nuevo `python3 pipeline_base.py` y comprueben que el MAE sube de `4.83` a `19.99` kWh.

---

## 6. Preguntas filtro para destrabar proyectos hoy mismo

Cuando pase por los lugares revisando las propuestas en `plantilla_propuesta.md`, haga estas cuatro preguntas:

1. **¿Ya tienes los datos hoy o dependes de que alguien te los dé en diciembre?** *(Si no tienen acceso seguro a los datos esta misma semana, deben cambiar a un dataset público o a un generador/simulador del curso).*
2. **¿Qué columna sería trampa incluir en $X$?** *(Verifica que entiendan fuga de datos en su propio problema).*
3. **¿Cuál es tu predicción "tonta" (*baseline*) contra la que vas a competir?** *(Si no pueden definir una referencia de 3 líneas de código, la tarea aún no está bien formulada).*
4. **¿Cómo vas a explicar cuando tu modelo se equivoque?** *(Recuerda que el 25 % de la nota del 15 de enero es la autopsia de los peores errores).*

