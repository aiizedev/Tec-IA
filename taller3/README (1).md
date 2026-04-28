# 🧠 Taller 3 — Diseño y Optimización de un MLP Profundo
### Técnicas de Inteligencia Artificial | Dataset: Olivetti Faces

---

## 📋 Descripción General

Se diseñó, entrenó y optimizó un **Perceptrón Multicapa (MLP)** para clasificación multiclase de identidades faciales sobre el dataset **Olivetti Faces** (400 imágenes, 40 clases, 64×64 px). Se implementó una **Búsqueda Aleatoria de 50 iteraciones** explorando tres perfiles arquitectónicos: Robusto, Profundo y Adaptativo.

---

## 📂 Estructura de Datos

| Subconjunto | Muestras | Porcentaje |
|---|:---:|:---:|
| Entrenamiento | 240 | 60% |
| Validación | 80 | 20% |
| Prueba | 80 | 20% |

> Partición con **muestreo estratificado** → 6 train / 2 val / 2 test por persona.

---

## 🏆 Resultados del Top 3

| Rank | Perfil | Test Accuracy | Observación |
|:---:|---|:---:|---|
| 🥇 1 | Robusto | **95.00%** | Convergencia rápida (~42 épocas) |
| 🥈 2 | Robusto | ~95.00% | Convergencia lenta (~150 épocas) |
| 🥉 3 | Robusto | ~90.00% | Inestabilidad en train loss |

> Los tres modelos del Top 3 pertenecen al perfil **Robusto** (ReLU + Dropout + Adam), lo que sugiere que la regularización por Dropout fue la estrategia más efectiva para este dataset pequeño.

---

## 📊 Parte 6: Discusión y Análisis

---

### 1. ¿Cuál fue el impacto de la profundidad?

El perfil **Robusto** dominó el Top 3 con arquitecturas de **2 capas ocultas** (configuraciones típicas: 1024→256 o 512→128 neuronas). Esto indica que para el dataset Olivetti — con solo 240 muestras de entrenamiento — una red de profundidad moderada es suficiente y preferible.

Aumentar la profundidad sin suficientes datos tiende a incrementar la varianza del modelo. Los perfiles **Profundo** (3-4 capas con SELU) y **Adaptativo** no lograron superar al Robusto, posiblemente porque las capas adicionales introducen más parámetros de los que el dataset puede sostener, incluso con regularización L2.

**Conclusión:** En datasets pequeños, la profundidad óptima es baja (2 capas). Más capas no implica mejor rendimiento si los datos son insuficientes.

---

### 2. ¿La regularización fue necesaria?

**Sí, fue fundamental.** El perfil ganador usó **Dropout (0.3–0.5)**, y su efectividad se evidencia directamente en las curvas de aprendizaje:

- **Rank 1:** Train Loss y Val Loss convergen juntas y de forma estable hacia valores cercanos a 0. Las curvas de accuracy muestran que la validación alcanza ~97% mientras el entrenamiento llega a ~99%, una diferencia mínima que indica generalización saludable.
- Sin Dropout, en un dataset de solo 240 muestras con 4096 features de entrada, el modelo memorizaría el conjunto de entrenamiento (accuracy 100% train, bajo en val).

El Dropout actuó como un "apagador aleatorio de neuronas" durante el entrenamiento, forzando a la red a aprender representaciones más robustas y no depender de neuronas específicas.

---

### 3. ¿Se observó sobreajuste?

El análisis de las curvas revela comportamientos distintos según el rank:

**Rank 1 — Sin sobreajuste significativo:**
Train Acc (~99%) y Val Acc (~97%) convergen de forma estable. El gap es pequeño y constante. El Dropout controló efectivamente el sobreajuste. La convergencia se logró en ~42 épocas, lo que indica que el Early Stopping actuó en el momento correcto.

**Rank 2 — Sobreajuste leve tardío:**
La Val Acc supera al Train Acc durante casi todo el entrenamiento (fenómeno propio del Dropout, que solo se desactiva en inferencia). Hacia las épocas finales (~130-150) las curvas se acercan pero con ligera inestabilidad. El modelo necesitó muchas más épocas para converger, lo que sugiere una tasa de aprendizaje más baja.

**Rank 3 — Sobreajuste moderado:**
Se observa una brecha más notoria entre Train Loss y Val Loss, especialmente en las primeras 20 épocas. La Val Loss cae rápido y se estabiliza, mientras el Train Loss sigue bajando lentamente. Esto es un patrón clásico de sobreajuste parcial, aunque el Early Stopping lo contuvo antes de que empeorara.

---

### 4. ¿Qué conclusiones pueden extraerse sobre la estructura de los datos?

**a) El dataset es linealmente separable en alta dimensión:**
Un MLP sin convoluciones alcanzó 95–97% de accuracy, lo que demuestra que las 4096 features (píxeles) contienen suficiente información discriminativa para separar las 40 identidades de forma casi perfecta con una transformación no lineal simple.

**b) Las clases con menor F1-score son las más similares visualmente:**
Las personas 2, 9 y 12 mostraron recall de 0.50 (una imagen mal clasificada de dos). Esto sugiere que esas identidades comparten características faciales más similares entre sí, o que sus dos imágenes de prueba presentan variaciones de iluminación o expresión que el modelo no generalizó bien.

**c) El dataset es pequeño pero bien estructurado:**
Con solo 6 imágenes por persona para entrenar, el modelo logró alta precisión gracias a la estratificación y la normalización Z-score. La estandarización fue crítica: sin ella, los gradientes serían inestables con 4096 features de entrada.

**d) MLP vs CNN:**
Si bien el MLP alcanzó excelentes resultados, una CNN aprovecharía la estructura espacial 64×64 de las imágenes (bordes, texturas, formas), probablemente superando el 97% con menos parámetros. El MLP trata cada píxel de forma independiente, perdiendo información de vecindad.

---

## 📈 Métricas Finales del Mejor Modelo (Rank 1 — Robusto)

| Métrica | Valor |
|---|:---:|
| Test Accuracy | **95.00%** |
| Macro Precision | 0.98 |
| Macro Recall | 0.97 |
| Macro F1-score | 0.97 |
| Clases con error | 3 de 40 (personas 2, 9, 12) |

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-green) ![NumPy](https://img.shields.io/badge/NumPy-2.x-blue) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-red)
