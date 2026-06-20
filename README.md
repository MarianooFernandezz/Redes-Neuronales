# Trabajo Práctico Final - Matemática III

**Integrantes:**
* Fernandez Mariano
* Moro Martín

---

## Objetivo: Red Neuronal
Su objetivo principal es predecir si un paciente puede sufrir un paro cardíaco basándose en parámetros médicos.

---

## Parte 1: Análisis de la Base de Datos

### 1. Descripción del Conjunto de Datos
A continuación se detallan las características de las variables analizadas:

* **Age (Edad):** Representa la edad de los pacientes. Es una variable continua.
* **Sex (Sexo):** Describe el género de la persona. Es una variable binaria.
* **ExerciseAngina (Angina inducida por ejercicio):** Expresa la presencia o no de angina inducida por el ejercicio. Variable binaria (`1`: Presencia, `0`: Ausencia).
* **ChestPainType (Tipo de dolor de pecho):** Variable categórica que representa el tipo de dolor:
  * `1`: Angina típica
  * `2`: Angina atípica
  * `3`: Dolor no anginal
  * `4`: Asintomático
* **RestingBP (Presión arterial en reposo):** Mide la presión arterial en mm Hg. Variable continua.
* **Cholesterol (Colesterol):** Mide el colesterol en mg/dl. Variable continua.
* **FastingBS (Azúcar en ayunas):** Variable binaria (`1` indica nivel $> 120$ mg/dl).
* **RestingECG (Resultados del electrocardiograma en reposo):** Variable categórica:
  * `0`: Normal
  * `1`: Anormalidad en la onda ST-T
  * `2`: Hipertrofia ventricular izquierda probable o definitiva (Criterio de Estes)
* **MaxHR (Frecuencia cardíaca máxima):** Variable continua.
* **ST_Slope (Pendiente ST):** Representa la pendiente del segmento ST. Variable categórica:
  * `0`: Descendente
  * `1`: Plana
  * `2`: Ascendente
* **Oldpeak:** Depresión del segmento ST medida en milímetros. Variable numérica continua.
* **HeartDisease (Variable Objetivo):** Variable de salida binaria:
  * `0`: Menor probabilidad de ataque cardíaco
  * `1`: Mayor probabilidad de ataque cardíaco

### 2. Análisis de Correlaciones
El mapa de correlaciones revela los distintos grados de influencia de las características con respecto a la probabilidad de ataque cardíaco:
* **Correlación moderada (relación apreciable):** `ChestPainType`, `MaxHR`, `ExerciseAngina`, `Oldpeak` y `ST_Slope`.
* **Correlación débil (pero útiles para el modelo):** `Age`, `Sex` y `FastingBS`.
* **Correlación muy débil:** `RestingBP`, `Cholesterol` y `RestingECG`.

### 3. Análisis de Factibilidad
La base de datos es **adecuada** para entrenar una red neuronal de clasificación binaria debido a:
* **Cantidad y calidad de datos:** 918 registros iniciales son suficientes para evitar sobreajuste en una red sencilla.
* **Cantidad de características:** Las variables disponibles están íntimamente relacionadas con la salud cardíaca.
* **Propósito del modelo:** Al tener una salida binaria (0 o 1), el entrenamiento de la red se optimiza y se vuelve más eficiente para su predicción.

### 4. Datos Atípicos (Outliers) y Limpieza
Se identificaron valores atípicos en las variables numéricas continuas (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`). Por tratarse de un contexto de salud, se interpretaron como posibles errores de medición o valores irrelevantes para el patrón general y se procedió a eliminarlos, reduciendo la muestra a **702 registros**.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seleccionar las columnas numéricas para hacer el análisis
numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Creamos los boxplots para visualizar los datos atípicos
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(df[col])
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.show()

# Función para eliminar datos atípicos (usando el rango intercuartílico)
def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Eliminar outliers
df = remove_outliers(df, numeric_columns)
```

### 5. Transformaciones Preliminares
* **Normalización:** Se estandarizaron los datos para asegurar que la red neuronal procese rangos de valores similares, mejorando la velocidad de convergencia y el aprendizaje.
* **Codificación:** Se codificaron a formato numérico variables categóricas.
* *Nota:* No fue necesaria la eliminación de valores nulos ya que el dataset estaba íntegro.

---

## Parte 2: Desarrollo de la Red Neuronal

### 1. Arquitectura de la Red
* **Capas y Neuronas:** La red consta de una capa de entrada (11 variables), una **capa oculta (9 neuronas)** y una **capa de salida (1 neurona)**.
* **Funciones de Activación:**
  * *Capa Oculta:* **ReLU**, elegida por su alta velocidad de convergencia y bajo costo computacional.
  * *Capa de Salida:* **Logística (Sigmoide)**, ideal para clasificación binaria, donde salidas $\ge 0.5$ indican alto riesgo de infarto.

### 2. Análisis de Overfitting (Sobreajuste)
Evaluando los gráficos de precisión (Accuracy) para entrenamiento y testeo, **el modelo final no presenta overfitting**. Durante el desarrollo sí se experimentó este problema (curvas separándose), el cual se solucionó calibrando:
* La cantidad de iteraciones (steps).
* La tasa de aprendizaje (learning rate).
* La cantidad de neuronas en la capa oculta.

---

## Parte 3: Comparación con `scikit-learn`

Al comparar la red neuronal implementada analíticamente en `numpy` frente a la herramienta estándar `scikit-learn`, destacaron los siguientes aspectos:
* **Similitudes:** Tras un ajuste adecuado, la implementación manual logró resultados de precisión muy parecidos a los entregados por `scikit-learn`.
* **Diferencias:** La versión de `scikit-learn` está altamente optimizada, por lo que logra converger utilizando muchas menos iteraciones y en menor tiempo de ejecución.

---

## Parte 4: Conclusiones Finales

El ejercicio de construir la red desde cero consolidó el entendimiento estructural y matemático subyacente al Deep Learning (multiplicación matricial, funciones de costo, descenso de gradiente y propagación).

| Método | Ventajas | Desventajas |
| :--- | :--- | :--- |
| **Manual (NumPy)** | Control absoluto del código y los parámetros. Facilidad para graficar paso a paso y evaluar el *overfitting* de forma transparente. | Alta complejidad; mayor propensión a errores lógicos (bugs) y un tiempo de ejecución considerablemente más largo. |
| **`scikit-learn`** | Código breve, sencillo, rápido y muy optimizado. | Opera como "caja negra" (menor visibilidad del proceso) y no provee herramientas integradas fáciles para graficar curvas de aprendizaje paso a paso. |

**Reflexión:** Ambas metodologías son altamente complementarias. Las implementaciones manuales brindan bases sólidas de conocimiento, mientras que herramientas como `scikit-learn` potencian la eficiencia requerida para producción.
