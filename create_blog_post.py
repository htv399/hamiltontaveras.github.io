import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Función para convertir gráficos a imágenes base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Cargar los datos
data = load_iris()
X = data.data
y = data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el Modelo
accuracy = accuracy_score(y_test, y_pred)

# Crear gráficos de dispersión y convertirlos a base64
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', ax=axes[0])
axes[0].set_title('Sepal Length vs Sepal Width')
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')

sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y, palette='viridis', ax=axes[1])
axes[1].set_title('Petal Length vs Petal Width')
axes[1].set_xlabel('Petal Length')
axes[1].set_ylabel('Petal Width')

plt.tight_layout()

# Convertir el gráfico completo a una imagen base64
scatter_base64 = fig_to_base64(fig)
plt.close(fig)

# Configura las variables para tu nueva entrada
title = "Random Forest Estimator"
author = "MA. Hamilton Taveras"
description = "Una breve descripción de tu entrada."
content = """
## ¿Qué es el Modelo Random Forest?

El Random Forest es un método de aprendizaje supervisado que se utiliza tanto para clasificación como para regresión. Funciona creando múltiples árboles de decisión durante el entrenamiento y sacando la media de las predicciones de estos árboles para obtener un resultado más preciso y robusto.

### ¿Cómo Funciona?

1. **Selección de Muestras**: De los datos de entrenamiento, se seleccionan múltiples muestras aleatorias con reemplazo.
2. **Construcción de Árboles de Decisión**: Para cada muestra, se construye un árbol de decisión. Cada árbol es entrenado utilizando diferentes subconjuntos de características.
3. **Agregación de Resultados**: Para la clasificación, se toma el modo de las predicciones de todos los árboles. Para la regresión, se toma el promedio.

### Implementación en Python

A continuación, te mostramos cómo implementar un modelo Random Forest utilizando la biblioteca `scikit-learn`.

### Visualización de la data


```python
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y, palette='viridis')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.tight_layout()
plt.show()
```

![Scatter Plot](data:image/png;base64,{scatter_base64})

#### Paso 1: Importar las Librerías Necesarias

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

#### Paso 2: Cargar los datos
```python
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
```

#### Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Paso 4: Crear el modelo Random Forest

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

#### Paso 5: Entrenar el modelo

```python
model.fit(X_train, y_train)
```

#### Paso 6: Realizar predicciones

```python
y_pred = model.predict(X_test)
```

#### Paso 7: Evaluar el Modelo

```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
```
### Conclusión
El modelo Random Forest es una herramienta poderosa y versátil para tareas de clasificación y regresión. Su capacidad para manejar grandes conjuntos de datos y su resistencia al sobreajuste lo hacen una opción popular entre los científicos de datos.

"""

# Formatea la fecha
date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# Crea el contenido del archivo
post_content = f"""\
---
title: "{title}"
date: {date}
author: "{author}"
description: "{description}"
---

{content}
"""

# Define la ruta donde se guardará la entrada
post_filename = f"{date[:10]}-{title.lower().replace(' ', '-')}.md"
post_filepath = os.path.join("content", "posts", post_filename)

# Crea el directorio si no existe
os.makedirs(os.path.dirname(post_filepath), exist_ok=True)

# Escribe el contenido al archivo
with open(post_filepath, "w", encoding="utf-8") as file:
    file.write(post_content)

print(f"Entrada de blog creada en: {post_filepath}")