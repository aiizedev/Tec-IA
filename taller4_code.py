import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#===========================================================
# Generar datos bidimensionales (2 características)
X, y = make_classification(n_samples=200, n_features=2,
                           n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, flip_y=0.05,
                           class_sep=1.5, random_state=0)

#===========================================================
#===========================================================
# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

#===========================================================
# Entrenar modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

#===========================================================
# Obtener coeficientes y mostrar ecuación de la frontera
w = model.coef_[0]
b = model.intercept_[0]
print(f"\n\nEcuación de la frontera de decisión: {w[0]:.3f}·x₁ + {w[1]:.3f}·x₂ + {b:.3f} = 0")


#===========================================================
# Seleccionar un punto de prueba y calcular su probabilidad
idx = np.random.randint(len(X_test))
point = X_test[idx].reshape(1, -1)
prob = model.predict_proba(point)[0, 1]
print(f"\nPunto resaltado: x = {point[0]}, Probabilidad estimada de clase 1 = {prob:.3f}")
print("\n\n")

#===========================================================
# Graficar datos y frontera de decisión
plt.figure(figsize=(8, 6))

# Malla para el fondo
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = -2, 6  # Ajuste de los límites del eje y
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

#===========================================================
# Predicción sobre cada punto de la malla
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#===========================================================
# Dibujar región de decisión
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

#===========================================================
# Dibujar puntos reales
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
            cmap=plt.cm.RdBu, edgecolors='k', label='Entrenamiento')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
            cmap=plt.cm.RdBu, edgecolors='gray', marker='x', label='Prueba')

#===========================================================
# Dibujar la frontera de decisión (línea)
x_vals = np.linspace(x_min, x_max, 300)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Frontera de decisión')

#===========================================================
# Dibujar punto resaltado
plt.scatter(point[0, 0], point[0, 1], color='gold', edgecolor='black',
            s=150, marker='o', label=f'Punto resaltado\nP(y=1)={prob:.2f}')

#===========================================================
# Configurar gráfico
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.ylim(-2, 6)  # Establecer el eje Y de -2 a 6
plt.title('Regresión Logística en 2D con punto resaltado')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification


#=========================================================================
# Generar datos de dos clases en 2D (al menos 20 puntos por clase)
X, y = make_classification(n_samples=50, n_features=2,
                           n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=1.5,
                           random_state=42)


#=========================================================================
#  Entrenar SVM lineal con margen suave
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)


#=========================================================================
#  Extraer coeficientes
w = clf.coef_[0]
b = clf.intercept_[0]


#=========================================================================
#  Calcular margen
margin = 1 / np.linalg.norm(w)


#=========================================================================
#  Crear malla para graficar frontera de decisión
xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500)
yy = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)


#=========================================================================
#  Obtener todos los vectores de soporte
support_vectors = clf.support_vectors_


#=========================================================================
#  Filtrar solo los vectores de soporte exactamente en el margen (±1)
decision_values = clf.decision_function(support_vectors)
epsilon = 1e-3  # tolerancia
on_margin_mask = np.isclose(np.abs(decision_values), 1.0, atol=epsilon)
margin_support_vectors = support_vectors[on_margin_mask]



#=========================================================================
#  Graficar
plt.figure(figsize=(8, 6))


# --- Eliminar los puntos que están dentro del margen pero son vectores de soporte ---
# Índices de todos los vectores de soporte
all_sv_indices = clf.support_

# Índices de los vectores de soporte sobre el margen
margin_sv_indices = all_sv_indices[on_margin_mask]

# Crear una máscara booleana para todos los puntos
mask = np.ones(len(X), dtype=bool)
mask[all_sv_indices] = False  # eliminar todos los vectores de soporte
mask[margin_sv_indices] = True  # conservar solo los que están sobre el margen

# Graficar puntos (excluyendo los SV internos al margen)
plt.scatter(X[mask, 0], X[mask, 1], c=y[mask], cmap='bwr', s=30, edgecolors='k')


## Graficar todos los puntos
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30, edgecolors='k')

# Frontera de decisión y márgenes
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
            linestyles=['--', '-', '--'])

# Añadir etiquetas con las ecuaciones de las márgenes y la frontera
x_pos = X[:, 0].mean()

# Línea +1 (margen positivo)
y_plus = (-w[0]*x_pos - b + 1)/w[1]
plt.text(x_pos + 0.2, y_plus, r'$w^T x + b = +1$', fontsize=12, color='green')

# Línea -1 (margen negativo)
y_minus = (-w[0]*x_pos - b - 1)/w[1]
plt.text(x_pos + 0.2, y_minus, r'$w^T x + b = -1$', fontsize=12, color='purple')

# Línea de decisión
y_zero = (-w[0]*x_pos - b)/w[1]
plt.text(x_pos + 0.2, y_zero, r'$w^T x + b = 0$', fontsize=12, color='blue')

# Graficar solo los vectores de soporte que están exactamente en el margen
plt.scatter(margin_support_vectors[:, 0], margin_support_vectors[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Vectores sobre el margen')

# Otra frontera de decisión con menor margen (ejemplo: pendiente distinta)
x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
plt.plot(x_vals, -0.5 * x_vals + 1, 'b--', label='Otra frontera de decisión con menor margen')

plt.title("SVM: Frontera de decisión y margen")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.grid(True)
plt.axis('tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification


##########################################################################
# Crear datos bidimensionales no linealmente separables perfectos
X, y = make_classification(n_samples=40, n_features=2,
                           n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=0.8,
                           random_state=42)
y = 2 * y - 1  # Convertir etiquetas a -1 y 1


##########################################################################
# Ajustar dos clasificadores SVM con diferentes valores de C
Cs = [0.1, 100]  # C pequeño vs C grande

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, C in zip(axes, Cs):
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X, y)

    # Obtener w y b para calcular márgenes
    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin = 1 / np.linalg.norm(w)

    # Recta frontera: w^T x + b = 0
    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500)
    yy = -(w[0] * xx + b) / w[1]

    # Márgenes: w^T x + b = ±1
    yy_margin1 = -(w[0] * xx + b - 1) / w[1]
    yy_margin2 = -(w[0] * xx + b + 1) / w[1]

    # Graficar puntos
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # Dibujar frontera de decisión y márgenes
    ax.plot(xx, yy, 'k-', label=r'$\mathbf{w}^T\mathbf{x} + b = 0$')
    ax.plot(xx, yy_margin1, 'k--', label=r'Margen $+1$')
    ax.plot(xx, yy_margin2, 'k--', label=r'Margen $-1$')

    # Dibujar vectores de soporte
    ax.scatter(clf.support_vectors_[:, 0],
               clf.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', label='Vectores soporte')

    ax.set_title(f"SVM con C = {C}")
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.legend()
    ax.grid(True)

plt.suptitle("Comparación del efecto de C en SVM", fontsize=14)
plt.tight_layout()
plt.show()

#========================================================================
# Ejemplo: SVM Lineal vs SVM con Kernel RBF
# Regiones de decisión coloreadas
#========================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

#---------------------------------------------------------
# 1. Generar datos no linealmente separables (dos círculos)
#---------------------------------------------------------

X, y = make_circles(n_samples=400, factor=0.4, noise=0.08, random_state=0)

#---------------------------------------------------------
# 2. Entrenar modelos
#---------------------------------------------------------

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X, y)

svm_rbf = SVC(kernel='rbf', gamma=2, C=1)
svm_rbf.fit(X, y)

#---------------------------------------------------------
# 3. Crear malla para regiones de decisión
#---------------------------------------------------------

h = 0.01

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]

Z_linear = svm_linear.predict(grid)
Z_linear = Z_linear.reshape(xx.shape)

Z_rbf = svm_rbf.predict(grid)
Z_rbf = Z_rbf.reshape(xx.shape)

#---------------------------------------------------------
# 4. Graficar
#---------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(12,5))

#---------------------------
# SVM Lineal
#---------------------------

ax[0].contourf(xx, yy, Z_linear, alpha=0.3, cmap=plt.cm.coolwarm)
ax[0].scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
ax[0].set_title("SVM con Kernel Lineal")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")

#---------------------------
# SVM RBF
#---------------------------

ax[1].contourf(xx, yy, Z_rbf, alpha=0.3, cmap=plt.cm.coolwarm)
ax[1].scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
ax[1].set_title("SVM con Kernel RBF")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles
from matplotlib.colors import ListedColormap

# 1. Generar datos no linealmente separables
X, y = make_circles(n_samples=200, factor=0.3, noise=0.05, random_state=0)

# 2. Definir dos clasificadores SVM
clf_linear = svm.SVC(kernel='linear', C=1.0)
clf_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

# 3. Entrenar los clasificadores
clf_linear.fit(X, y)
clf_rbf.fit(X, y)

# 4. Crear una malla para graficar las fronteras
h = .02  # paso de la malla
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 5. Predecir sobre la malla
Z_linear = clf_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])

Z_linear = Z_linear.reshape(xx.shape)
Z_rbf = Z_rbf.reshape(xx.shape)

# 6. Graficar
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

titles = ['SVM Lineal', 'SVM con Núcleo RBF']
models = [Z_linear, Z_rbf]

for ax, Z, title in zip(axes, models, titles):
    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue']), s=30, edgecolors='k')
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


#========================================================================
# Crear conjunto de datos no linealmente separable
X, y = make_moons(n_samples=100, noise=0.25, random_state=42)


#========================================================================
# Crear función para graficar regiones de decisión
def plot_knn_decision_boundary(k_values, X, y):
    cmap_light = ListedColormap(['#FFBBBB', '#BBBBFF'])
    cmap_bold = ['red', 'blue']

    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 4))

    for ax, k in zip(axes, k_values):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)

        # Crear malla de puntos para dibujar frontera
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        for i, color in enumerate(cmap_bold):
            ax.scatter(X[y == i, 0], X[y == i, 1], c=color, label=f'Clase {i}', edgecolor='k')

        ax.set_title(f'k = {k}')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.legend(loc='upper right')

    plt.suptitle('Efecto del valor de $k$ en k-NN')
    plt.tight_layout()
    plt.show()

# 3. Ejecutar con distintos valores de k
plot_knn_decision_boundary([1, 5, 30], X, y)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


#========================================================================
#  Generar datos no lineales con estructura circular
X, y = make_circles(n_samples=300, noise=0.2, factor=0.4, random_state=0)


#========================================================================
#  Función para graficar la frontera de decisión
def plot_knn_boundaries_metric(X, y, k_values, metrics):
    cmap_light = ListedColormap(['#FFCCCC', '#CCCCFF'])
    cmap_bold = ['red', 'blue']

    fig, axes = plt.subplots(len(metrics), len(k_values), figsize=(15, 6))

    for i, metric in enumerate(metrics):
        for j, k in enumerate(k_values):
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
            clf.fit(X, y)

            # Malla para frontera
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                 np.linspace(y_min, y_max, 300))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            ax = axes[i, j]
            ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
            for class_value, color in enumerate(cmap_bold):
                ax.scatter(X[y == class_value, 0], X[y == class_value, 1],
                           c=color, edgecolor='k', label=f'Clase {class_value}', s=20)
            ax.set_title(f'k = {k}, Métrica: {metric}')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Influencia de $k$ y la métrica en k-NN', fontsize=14)
    plt.tight_layout()
    plt.show()

# 3. Ejecutar para k pequeño y grande, y dos métricas
plot_knn_boundaries_metric(X, y, k_values=[1, 20], metrics=['euclidean', 'manhattan'])


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


#========================================================================
#  Generar datos dispersos y no lineales
X, y = make_moons(n_samples=80, noise=0.3, random_state=42)


#========================================================================
#  Función para graficar fronteras
def plot_knn_boundaries_metric(X, y, k_values, metrics):
    cmap_light = ListedColormap(['#FFCCCC', '#CCCCFF'])
    cmap_bold = ['red', 'blue']

    fig, axes = plt.subplots(len(metrics), len(k_values), figsize=(12, 6))

    for i, metric in enumerate(metrics):
        for j, k in enumerate(k_values):
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
            clf.fit(X, y)

            # Crear malla para visualizar frontera
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                 np.linspace(y_min, y_max, 300))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            ax = axes[i, j]
            ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
            for class_value, color in enumerate(cmap_bold):
                ax.scatter(X[y == class_value, 0], X[y == class_value, 1],
                           c=color, edgecolor='k', label=f'Clase {class_value}', s=30)
            ax.set_title(f'k = {k}, Métrica: {metric}')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Fronteras de decisión: efecto de $k$ y la métrica', fontsize=14)
    plt.tight_layout()
    plt.show()


#========================================================================
#  Probar con k pequeño y grande, y dos métricas
plot_knn_boundaries_metric(X, y, k_values=[1, 10], metrics=['euclidean', 'manhattan'])


# Árbol de decisión para clasificación
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report


#==============================================================================
# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target


#==============================================================================
# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#==============================================================================
#  Crear y entrenar el árbol de decisión
clf = DecisionTreeClassifier(
    criterion="entropy",   # también se puede usar "gini"
    max_depth=3,           # limitar la profundidad para simplificar
    random_state=42
)
clf.fit(X_train, y_train)


#==============================================================================
#  Predicciones
y_pred = clf.predict(X_test)


#==============================================================================
#  Evaluación del modelo
print("Exactitud:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


#==============================================================================
#  Visualizar el árbol
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, rounded=True)
plt.show()


# Random Forest con Iris Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Crear modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100,   # número de árboles
    max_depth=3,        # profundidad máxima de cada árbol
    random_state=42
)

# Entrenar
rf.fit(X_train, y_train)

# Predicciones
y_pred = rf.predict(X_test)

# Evaluación
print("Exactitud:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Importancia de características
importances = rf.feature_importances_
features = iris.feature_names

# Mostrar importancias en gráfico
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Importancia de características en Random Forest")
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Valores de número de estimadores a probar
n_estimators_list = [1, 5, 10, 20, 50, 100, 200]

accuracies = []

# Entrenar y evaluar con diferentes n_estimators
for n in n_estimators_list:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Graficar resultados
plt.figure(figsize=(8,5))
plt.plot(n_estimators_list, accuracies, marker='o', linestyle='-')
plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Exactitud en prueba (Accuracy)")
plt.title("Desempeño del Random Forest en función del número de árboles")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar dataset de dígitos
digits = load_digits()
X, y = digits.data, digits.target

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Probar diferentes números de árboles
#estimadores = [10, 50, 100, 200, 300]
estimadores = [1, 5, 10, 20, 50, 100, 200]
accuracies = []

for n in estimadores:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Graficar desempeño vs número de árboles
plt.figure(figsize=(8,5))
plt.plot(estimadores, accuracies, marker='o', linestyle='--')
plt.xlabel("Número de árboles (estimadores)")
plt.ylabel("Exactitud en test")
plt.title("Random Forest en dataset Digits")
plt.show()

# Entrenar un modelo final con más árboles
clf_final = RandomForestClassifier(n_estimators=200, random_state=42)
clf_final.fit(X_train, y_train)

# Importancia de características (píxeles)
importances = clf_final.feature_importances_
importances_img = importances.reshape(8, 8)

plt.figure(figsize=(6,6))
plt.imshow(importances_img, cmap="hot", interpolation="nearest")
plt.title("Importancia de los píxeles en Random Forest")
plt.colorbar()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------
# 1. Cargar dataset
# ------------------------------
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names

# ------------------------------
# 2. Dividir datos en entrenamiento y prueba
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------
# 3. Evaluar desempeño según n_estimators
# ------------------------------
estimators = [1, 5, 15, 30, 40, 50, 100]
accuracies = []

for n in estimators:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# ------------------------------
# 4. Entrenar modelo final con más árboles
# ------------------------------
clf_final = RandomForestClassifier(n_estimators=200, random_state=42)
clf_final.fit(X_train, y_train)

# ------------------------------
# 5. Importancia de características
# ------------------------------
importances = clf_final.feature_importances_
indices = np.argsort(importances)[::-1]  # ordenar descendentemente

# ------------------------------
# 6. Graficar resultados
# ------------------------------

# Precisión vs número de árboles
plt.figure(figsize=(12,5))

plt.plot(estimators, accuracies, marker="o", linestyle="--")
plt.title("Precisión vs Número de Estimadores")
plt.xlabel("Número de árboles")
plt.ylabel("Precisión")
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report


#==============================================================================
# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target


#==============================================================================
# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#==============================================================================
# Crear y entrenar el modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100,     # número de árboles en el bosque
    criterion="gini",     # o "entropy"
    max_depth=3,          # limitar profundidad para visualización
    random_state=42
)

rf.fit(X_train, y_train)


#==============================================================================
# Predicciones
y_pred = rf.predict(X_test)


#==============================================================================
# Evaluación del modelo
print("Exactitud:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n",
      classification_report(y_test, y_pred, target_names=iris.target_names))


#==============================================================================
# Visualizar uno de los árboles del Random Forest
plt.figure(figsize=(12,8))

plot_tree(
    rf.estimators_[0],   # primer árbol del bosque
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.title("Uno de los árboles del Random Forest")
plt.show()
# Obtener importancia de características
importances = rf.feature_importances_

# Ordenar características por importancia
indices = importances.argsort()[::-1]

# Nombres de características
feature_names = iris.feature_names

# Gráfica
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.title("Importancia de las características")
plt.tight_layout()
plt.show()

# Random Forest con dataset Wine
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#==============================================================================
# Cargar dataset Wine
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

#==============================================================================
# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#==============================================================================
# Entrenar Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

#==============================================================================
# Predicciones
y_pred = rf.predict(X_test)

#==============================================================================
# Evaluación
print("Exactitud:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n",
      classification_report(y_test, y_pred, target_names=wine.target_names))

#==============================================================================
# Importancia de características
importances = rf.feature_importances_
print("IMPORTANCIAS:",importances)
indices = importances.argsort()[::-1]

#==============================================================================
# Gráfica de importancia
plt.figure(figsize=(10,6))

plt.bar(range(X.shape[1]), importances[indices], align="center")

plt.xticks(
    range(X.shape[1]),
    [feature_names[i] for i in indices],
    rotation=90
)

plt.title("Importancia de las características (Random Forest - Wine)")
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm


#====================================================================================
# Generar datos de clasificación binaria
#X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
#                           n_redundant=0, n_clusters_per_class=1, random_state=42)

#====================================================================================
#  Crear conjunto de datos no linealmente separable
X, y = make_moons(n_samples=100, noise=0.25, random_state=42)


#====================================================================================
#  Definir modelo GBM (pocos árboles para ilustrar)
gbm = GradientBoostingClassifier(n_estimators=6, learning_rate=0.5,
                                 max_depth=2, random_state=42)


#====================================================================================
#  Entrenar modelo
gbm.fit(X, y)


#====================================================================================
#  Crear malla para visualizar la frontera de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))


#====================================================================================
#  Iterar sobre las etapas de boosting
for i, y_pred in enumerate(gbm.staged_predict(np.c_[xx.ravel(), yy.ravel()]), start=1):
    Z = y_pred.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))

    # Datos
    plt.scatter(X[y==0, 0], X[y==0, 1], c="red", edgecolor="k", label="Clase 0")
    plt.scatter(X[y==1, 0], X[y==1, 1], c="blue", edgecolor="k", label="Clase 1")

    # Etiquetas
    plt.title(f"GBM - Iteración {i}")
    plt.legend()
    plt.tight_layout()

    # Guardar figura de cada paso
    plt.savefig(f"GBM_step_{i}.png", dpi=300)
    plt.show()


#====================================================================================
#  También puedes mostrar todas las etapas en una sola figura
#fig, axes = plt.subplots(1, gbm.n_estimators, figsize=(15, 3))
#for ax, (i, y_pred) in zip(axes, enumerate(gbm.staged_predict(np.c_[xx.ravel(), yy.ravel()]), start=1)):
#    Z = y_pred.reshape(xx.shape)
#    ax.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
#    ax.scatter(X[y==0, 0], X[y==0, 1], c="red", edgecolor="k", s=20)
#    ax.scatter(X[y==1, 0], X[y==1, 1], c="blue", edgecolor="k", s=20)
#    ax.set_title(f"Iter {i}")
#    #ax.set_xticks([]); ax.set_yticks([])
#
#    # mostrar valores en los ejes
#    plt.tight_layout()

# Número máximo de columnas por fila
ncols = 3
nrows = int(np.ceil(gbm.n_estimators / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3*nrows))

# Asegurar que axes sea 1D para iterar fácilmente
axes = axes.flatten()

for ax, (i, y_pred) in zip(axes, enumerate(gbm.staged_predict(np.c_[xx.ravel(), yy.ravel()]), start=1)):
    Z = y_pred.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X[y==0, 0], X[y==0, 1], c="red", edgecolor="k", s=20)
    ax.scatter(X[y==1, 0], X[y==1, 1], c="blue", edgecolor="k", s=20)
    ax.set_title(f"Iter {i}")
    # ax.set_xticks([]); ax.set_yticks([])

# Quitar los ejes sobrantes si no se llenan todas las casillas
for j in range(i, len(axes)):
    fig.delaxes(axes[j])

plt.savefig("GBM_all_iterations.png", dpi=300)
plt.show()


#====================================================================================
#  Número de árboles débiles en el modelo
n_estimators = len(gbm.estimators_)


#====================================================================================
#  Determinar layout de subplots (ej. cuadrícula)
n_cols = 3  # número de columnas que quieras
n_rows = int(np.ceil(n_estimators / n_cols))


#====================================================================================
#  Crear figura con subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))


#====================================================================================
#  Asegurar que axes sea 1D para iterar fácilmente
axes = axes.ravel()


#====================================================================================
#  Crear paleta de colores
colors = cm.tab20(np.linspace(0, 1, n_estimators))

for i, stage in enumerate(gbm.estimators_):
    ax = axes[i]

#    # Fondo coloreado según la predicción final del ensamble completo
#    ax.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))

    # Predicción del árbol débil i
    stage_pred = stage[0].predict(np.c_[xx.ravel(), yy.ravel()])
    stage_pred = stage_pred.reshape(xx.shape)

    # Dibujar frontera del árbol i
    ax.contour(xx, yy, stage_pred,
               colors=[colors[i]], alpha=0.8, linewidths=1.2)

    ## Dibujar los puntos de datos
    #ax.scatter(X[y==0, 0], X[y==0, 1], c="red", edgecolor="k", label="Clase 0", s=30)
    #ax.scatter(X[y==1, 0], X[y==1, 1], c="blue", edgecolor="k", label="Clase 1", s=30)

    ax.set_title(f"Iteración {i+1}")


#====================================================================================
#  Eliminar subplots vacíos si sobran
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#====================================================================================
#  Número de árboles débiles en el modelo
n_estimators = len(gbm.estimators_)


#====================================================================================
#  Determinar layout de subplots (ej. cuadrícula)
n_cols = 3  # máximo 3 columnas
n_rows = int(np.ceil(n_estimators / n_cols))


#====================================================================================
#  Crear figura con subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.ravel()  # Asegurar que sea 1D


#====================================================================================
#  Colormap para las regiones (Clase 0 = rojo, Clase 1 = azul)
region_cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])


#====================================================================================
#  Paleta de colores para las fronteras
colors = cm.tab20(np.linspace(0, 1, n_estimators))


#====================================================================================
#  Recorrer las predicciones acumuladas (staged_predict)
for i, y_pred in enumerate(gbm.staged_predict(np.c_[xx.ravel(), yy.ravel()])):
    ax = axes[i]

    # Dar forma a la predicción acumulada
    stage_pred = y_pred.reshape(xx.shape)

    # Colorear regiones acumuladas hasta el árbol i
    ax.contourf(xx, yy, stage_pred, alpha=0.3, cmap=region_cmap)

    # Dibujar frontera del modelo acumulado
    ax.contour(xx, yy, stage_pred, levels=[0.5],
               colors=[colors[i]], alpha=0.9, linewidths=1.5)

    # Dibujar los puntos de datos
    ax.scatter(X[y==0, 0], X[y==0, 1], c="red", edgecolor="k", s=25, label="Clase 0" if i==0 else "")
    ax.scatter(X[y==1, 0], X[y==1, 1], c="blue", edgecolor="k", s=25, label="Clase 1" if i==0 else "")

    ax.set_title(f"Iteración {i+1}")


#====================================================================================
#  Eliminar subplots vacíos
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])


#====================================================================================
#  Leyenda única
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.1, 1))

plt.tight_layout()
plt.show()



#====================================================================================
#  Crear la figura
plt.figure(figsize=(8, 6))


#====================================================================================
# Fondo coloreado según la predicción final
plt.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))


#====================================================================================
#  Mostrar cada estimador intermedio (árbol débil)
for i, stage in enumerate(gbm.estimators_):
    # La predicción de cada árbol débil
    stage_pred = stage[0].predict(np.c_[xx.ravel(), yy.ravel()])
    stage_pred = stage_pred.reshape(xx.shape)
    plt.contour(xx, yy, stage_pred, colors="gray", alpha=0.3, linewidths=1)


#====================================================================================
#  Dibujar los datos
plt.scatter(X[y==0, 0], X[y==0, 1], c="red", edgecolor="k", label="Clase 0")
plt.scatter(X[y==1, 0], X[y==1, 1], c="blue", edgecolor="k", label="Clase 1")

# Añadir etiquetas
plt.title("Ilustración del Proceso de Gradient Boosting (Clasificación)")
plt.legend()
plt.tight_layout()

# Guardar la figura
plt.savefig("GBM_Classification.png", dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap


#===============================================================================
# Generar datos de ejemplo 2D
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           random_state=42)

#===============================================================================
#  Entrenar Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)

#===============================================================================
#  Crear malla para evaluar frontera de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#===============================================================================
#  Graficar frontera de decisión y puntos de entrenamiento
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_points = ListedColormap(['#FF0000', '#0000FF'])

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor='k', s=50)
plt.title("Visualización de la Frontera de Decisión - Gaussian Naive Bayes")
plt.xlabel("Atributo 1")
plt.ylabel("Atributo 2")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler, Binarizer

# ---------------------------
# Generar dataset sintético
# ---------------------------
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)

# Escalar X para multinomial y bernoulli (requiere datos >= 0)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Binarizar para BernoulliNB
binarizer = Binarizer(threshold=0.5)
X_binary = binarizer.fit_transform(X_scaled)

# ---------------------------
# Definir clasificadores
# ---------------------------
classifiers = {
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB()
}

# Datos a usar para cada clasificador
X_dict = {
    "GaussianNB": X,
    "MultinomialNB": X_scaled,
    "BernoulliNB": X_binary
}

# ---------------------------
# Crear malla para visualización
# ---------------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# ---------------------------
# Graficar fronteras de decisión
# ---------------------------
plt.figure(figsize=(15,5))

for i, (name, clf) in enumerate(classifiers.items(), 1):
    X_train = X_dict[name]
    clf.fit(X_train, y)

    # Predecir sobre la malla
    if name == "GaussianNB":
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:  # Multinomial y Bernoulli requieren datos escalados o binarios
        # Ajustar escala/binarización de la malla
        if name == "MultinomialNB":
            mesh = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        else:
            mesh = binarizer.transform(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = clf.predict(mesh)

    Z = Z.reshape(xx.shape)

    plt.subplot(1, 3, i)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=plt.cm.RdBu)
    plt.title(name)
    plt.xlabel("X1")
    plt.ylabel("X2")

plt.tight_layout()
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ================================
# Paso Dataset
# ================================
texts = [
    "gato juega en el jardin",
    "perro come hueso",
    "gato duerme en el sofa",
    "perro protege la casa",
    "gato y perro son amigos",
    "perro ladra fuerte",
    "gato salta alto"
]

labels = ["gato", "perro", "gato", "perro", "gato", "perro", "gato"]

# ================================
# Paso Vectorización (vocabulario automático)
# ================================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("Vocabulario:", vectorizer.get_feature_names_out())

# ================================
# Paso Entrenar Naive Bayes
# ================================
model = MultinomialNB(alpha=1.0)  # Laplace smoothing
model.fit(X, labels)

# ================================
# Paso Probabilidades a priori
# ================================
print("\nProbabilidades a priori:")
for clase, prob in zip(model.classes_, model.class_log_prior_):
    print(f"P({clase}) = {round(np.exp(prob), 3)}")

# ================================
# Paso Clasificar nuevo documento
# ================================
doc_test = ["gato persigue raton"]
X_test = vectorizer.transform(doc_test)

pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print("\nDocumento:", doc_test[0])
print("Predicción:", pred[0])

print("\nProbabilidades por clase:")
for clase, p in zip(model.classes_, probs[0]):
    print(f"{clase}: {p:.6f}")
# ================================
# Paso Clasificar nuevo documento
# ================================
doc_test = ["perro corre en el parque"]
X_test = vectorizer.transform(doc_test)

pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print("\nDocumento:", doc_test[0])
print("Predicción:", pred[0])

print("\nProbabilidades por clase:")
for clase, p in zip(model.classes_, probs[0]):
    print(f"{clase}: {p:.6f}")
# ================================
# Paso Clasificar nuevo documento
# ================================
doc_test = ["perro juega con pelota"]
X_test = vectorizer.transform(doc_test)

pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print("\nDocumento:", doc_test[0])
print("Predicción:", pred[0])

print("\nProbabilidades por clase:")
for clase, p in zip(model.classes_, probs[0]):
    print(f"{clase}: {p:.6f}")
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# ================================
# Paso Dataset
# ================================
texts = [
    "gato juega en el jardin",
    "perro come hueso",
    "gato duerme en el sofa",
    "perro protege la casa",
    "gato y perro son amigos",
    "perro ladra fuerte",
    "gato salta alto"
]

labels = ["gato", "perro", "gato", "perro", "gato", "perro", "gato"]

# ================================
# Paso Vectorización binaria
# ================================
vectorizer = CountVectorizer(binary=True)  # 👈 clave: presencia/ausencia
X = vectorizer.fit_transform(texts)

print("Vocabulario:", vectorizer.get_feature_names_out())
print("\nMatriz binaria (0 = no aparece, 1 = aparece):\n", X.toarray())

# ================================
# Paso Entrenar modelo BernoulliNB
# ================================
model = BernoulliNB(alpha=1.0)  # Laplace smoothing
model.fit(X, labels)

# ================================
# Paso Probabilidades a priori
# ================================
print("\nProbabilidades a priori:")
for clase, logp in zip(model.classes_, model.class_log_prior_):
    print(f"P({clase}) = {np.exp(logp):.3f}")

# ================================
# Paso Clasificación
# ================================
doc_test = ["gato persigue raton"]
X_test = vectorizer.transform(doc_test)

pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print("\nDocumento:", doc_test[0])
print("Predicción:", pred[0])

print("\nProbabilidades por clase:")
for clase, p in zip(model.classes_, probs[0]):
    print(f"{clase}: {p:.6f}")
# ================================
# Paso Clasificación
# ================================
doc_test = ["perro corre en el parque"]
X_test = vectorizer.transform(doc_test)

pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print("\nDocumento:", doc_test[0])
print("Predicción:", pred[0])

print("\nProbabilidades por clase:")
for clase, p in zip(model.classes_, probs[0]):
    print(f"{clase}: {p:.6f}")
# ================================
# Paso Clasificación
# ================================
doc_test = ["perro juega con pelota"]
X_test = vectorizer.transform(doc_test)

pred = model.predict(X_test)
probs = model.predict_proba(X_test)

print("\nDocumento:", doc_test[0])
print("Predicción:", pred[0])

print("\nProbabilidades por clase:")
for clase, p in zip(model.classes_, probs[0]):
    print(f"{clase}: {p:.6f}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#=======================================================================================
# Generar datos
np.random.seed(0)
X = np.random.uniform(-2, 6, size=(50, 1))  # valores de entrada en [−2, 6]
y = 3 * X.squeeze() + 2 + np.random.normal(0, 1, size=50)  # salida con ruido


#=======================================================================================
#  Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)


#=======================================================================================
#  Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)


#=======================================================================================
#  Mostrar ecuación del modelo
w = model.coef_[0]
b = model.intercept_
print(f"\n\nEcuación de la recta: y = {w:.2f}·x + {b:.2f}")


#=======================================================================================
#  Punto de prueba específico
x_nuevo = np.array([[2.5]])
y_pred = model.predict(x_nuevo)
print(f"\nPara x = {x_nuevo[0][0]:.2f}, el valor predicho es ŷ = {y_pred[0]:.2f}")
print(f"\n\n")


#=======================================================================================
#  Graficar
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Entrenamiento')
plt.scatter(X_test, y_test, color='red', label='Prueba')
plt.plot(X, model.predict(X), color='black', linewidth=2, label='Recta de regresión')

# Punto nuevo resaltado
plt.scatter(x_nuevo, y_pred, color='lime', s=100, edgecolor='black', label='Predicción para x = 2.5')
plt.text(x_nuevo, y_pred + 1, f"$\\hat{{y}} = {y_pred[0]:.2f}$", fontsize=12, color='green')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresión Lineal (1D)')
plt.legend()
plt.grid(True)
plt.ylim(-2, 20)  # Eje y entre -2 y 20
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


#=======================================================================================
#   Generar datos (relación no lineal)
np.random.seed(1)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)


#=======================================================================================
#   Modelos SVR lineal y RBF con menor C en lineal
svr_lin = SVR(kernel='linear', C=1, epsilon=0.1)  # C reducido para menos vectores de soporte
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.1)

y_lin = svr_lin.fit(X, y).predict(X)
y_rbf = svr_rbf.fit(X, y).predict(X)


#=======================================================================================
#   Punto nuevo a predecir
X_new = np.array([[2.5]])
y_new_lin = svr_lin.predict(X_new)
y_new_rbf = svr_rbf.predict(X_new)


#=======================================================================================
#   Graficar ambos modelos
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

def plot_svr(ax, X, y, y_pred, model, title, X_new, y_new):
    ax.scatter(X, y, color='lightgray', label='Datos de entrenamiento')
    ax.plot(X, y_pred, 'k', label='SVR (función de regresión)')
    ax.plot(X, y_pred + model.epsilon, 'k--', label=r'Margen $\varepsilon$')
    ax.plot(X, y_pred - model.epsilon, 'k--')

    # Mostrar solo los vectores de soporte que coinciden con datos originales
    for sv in model.support_:
        ax.scatter(X[sv], y[sv], facecolors='none', edgecolors='red', s=100)

    # Punto nuevo predicho
    ax.scatter(X_new, y_new, color='blue', s=100, marker='x', label='Predicción nueva')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

# SVR Lineal
plot_svr(axs[0], X, y, y_lin, svr_lin, 'SVR Lineal (C=1)', X_new, y_new_lin)

# SVR RBF
plot_svr(axs[1], X, y, y_rbf, svr_rbf, 'SVR No Lineal (Núcleo RBF)', X_new, y_new_rbf)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


#=======================================================================================
#   Generar datos de regresión ruidosos y dispersos
np.random.seed(0)
X = np.sort(5 * np.random.rand(30, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])


#=======================================================================================
#   Crear conjunto de prueba para visualizar la curva
X_test = np.linspace(0, 5, 500).reshape(-1, 1)


#=======================================================================================
#   Probar diferentes valores de k
k_values = [1, 5, 15]
colors = ['red', 'green', 'blue']

plt.figure(figsize=(12, 4))
for i, k in enumerate(k_values):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X_test)

    plt.subplot(1, len(k_values), i + 1)
    plt.scatter(X, y, color='black', label='Datos', edgecolors='k')
    plt.plot(X_test, y_pred, color=colors[i], lw=2, label=f'k={k}')
    plt.ylim(-1.5, 1.5)
    plt.title(f'Regresión KNN con k = {k}')
    plt.xlabel('x')
    if i == 0:
        plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.suptitle('Efecto del valor de $k$ en la Regresión KNN', fontsize=14)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


#=======================================================================================
#   Generar datos dispersos en 2D
np.random.seed(1)
X = np.random.rand(100, 2) * 4 - 2  # valores entre [-2, 2]
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.3, 100)


#=======================================================================================
#   Crear malla para graficar la superficie
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]


#=======================================================================================
#   Visualizar para diferentes valores de k
k_values = [1, 5, 15]
fig = plt.figure(figsize=(15, 4))

for i, k in enumerate(k_values):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)
    zz = model.predict(grid).reshape(xx.shape)

    ax = fig.add_subplot(1, len(k_values), i + 1, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], y, color='r', s=20, label='Datos')
    ax.set_title(f'Regresión KNN 2D (k={k})')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('y')
    ax.view_init(elev=25, azim=135)

plt.suptitle('Efecto del valor de $k$ en la regresión KNN bidimensional', fontsize=14)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
# Generación de datos
# ------------------------------
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(80)

# ------------------------------
# Gráfico inicial de los datos
# ------------------------------
plt.figure(figsize=(6,4))
plt.scatter(X, y, s=25, edgecolor="black", c="darkorange", label="Datos")
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio (normalizado)")
plt.title("Datos iniciales")
plt.legend()
plt.show()

# ------------------------------
# Primera partición ilustrativa
# ------------------------------
s = 2.5
R1 = (X.ravel() <= s)
R2 = (X.ravel() > s)
y_R1 = np.mean(y[R1])
y_R2 = np.mean(y[R2])

plt.figure(figsize=(6,4))
plt.scatter(X, y, s=25, edgecolor="black", c="darkorange", label="Datos")
plt.axvline(s, color="red", linestyle="--", label=f"Umbral s={s}")
plt.hlines(y_R1, xmin=0, xmax=s, colors="blue", linewidth=2, label=r"$\hat{y}_{R_1}$")
plt.hlines(y_R2, xmin=s, xmax=5, colors="green", linewidth=2, label=r"$\hat{y}_{R_2}$")
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio (normalizado)")
plt.title("Primera partición y promedios locales")
plt.legend()
plt.show()

# ------------------------------
# Árbol de regresión entrenado
# ------------------------------
Profundidad = 3
reg = DecisionTreeRegressor(max_depth=Profundidad)
reg.fit(X, y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = reg.predict(X_test)

plt.figure(figsize=(6,4))
plt.scatter(X, y, s=25, edgecolor="black", c="darkorange", label="Datos")
plt.plot(X_test, y_pred, color="cornflowerblue", linewidth=2, label="Árbol de regresión (prof=3)")
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio (normalizado)")
plt.title("Árbol de Decisión para Regresión")
plt.legend()
plt.show()


# ------------------------------
# Mostrar el árbol de decisión final
# ------------------------------
plt.figure(figsize=(12,6))
plot_tree(reg, feature_names=["Tamaño (m²)"], filled=True, rounded=True)
plt.title(f"Árbol de Decisión para Regresión (profundidad = {Profundidad})")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree


#=======================================================================================
#   Datos de ejemplo
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 0.5 - np.random.rand(16)  # ruido


#=======================================================================================
#   Entrenar árbol de regresión con profundidad 3
reg = DecisionTreeRegressor(max_depth=3)
reg.fit(X, y)


#=======================================================================================
#   Crear grilla para predecir
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = reg.predict(X_test)


#=======================================================================================
#   Graficar datos y regresión
plt.figure(figsize=(10,6))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="datos")
plt.plot(X_test, y_pred, color="black", linewidth=2, label="predicción")


#=======================================================================================
#   Recorrer nodos del árbol y graficar thresholds
n_nodes = reg.tree_.node_count
children_left = reg.tree_.children_left
children_right = reg.tree_.children_right
threshold = reg.tree_.threshold


#=======================================================================================
#   Función recursiva para recorrer y pintar thresholds según profundidad
def plot_thresholds(node_id=0, depth=0):
    if children_left[node_id] != children_right[node_id]:  # nodo interno
        thr = threshold[node_id]
        color = {0:"red", 1:"blue", 2:"green"}.get(depth, "gray")
        plt.axvline(x=thr, color=color, linestyle="--", linewidth=1,
                    label=f"Umbral d={depth+1}" if f"Umbral d={depth+1}" not in plt.gca().get_legend_handles_labels()[1] else "")
        # Llamar recursivamente en hijos
        plot_thresholds(children_left[node_id], depth+1)
        plot_thresholds(children_right[node_id], depth+1)

plot_thresholds()

plt.xlabel("X")
plt.ylabel("y")
plt.title("Árbol de Regresión con Umbrales por Profundidad")
plt.legend()
plt.show()

# Graficar el árbol de decisión
plt.figure(figsize=(12,6))
plot_tree(reg, filled=True, feature_names=["X"], rounded=True)
plt.title("Árbol de Decisión - Regresión (Profundidad 3)")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


#=======================================================================================
#   Datos de ejemplo
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))  # añadir ruido


#=======================================================================================
#   Definir árboles base (random forest simple)
n_estimators = 10
Profundidad = 3
estimators = []
for _ in range(n_estimators):
    sample_idx = rng.choice(len(X), len(X), replace=True)
    X_sample, y_sample = X[sample_idx], y[sample_idx]
    tree = DecisionTreeRegressor(max_depth=Profundidad)
    tree.fit(X_sample, y_sample)
    estimators.append(tree)


#=======================================================================================
#   Predicciones
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_preds = np.zeros((n_estimators, len(X_test)))

for i, tree in enumerate(estimators):
    y_preds[i] = tree.predict(X_test)

y_mean = y_preds.mean(axis=0)


#=======================================================================================
#   Graficar
plt.figure(figsize=(8, 6))
for i in range(n_estimators):
    plt.plot(X_test, y_preds[i], color="gray", alpha=0.5)

plt.plot(X_test, y_mean, color="blue", linewidth=2, label="Random Forest (promedio)")
plt.scatter(X, y, c="red", edgecolor="k", label="Datos de entrenamiento")
plt.legend()
plt.xlabel("Edad (años)")
plt.ylabel("Ingreso (miles $)")
plt.title("Random Forest como promedio de árboles")
plt.savefig("fig/Latex/RandomForestReg.png", dpi=300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#=======================================================================================
#   Generar datos sintéticos (función no lineal + ruido)
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)   # datos en [0,5]
y_true_function = np.sin(X).ravel()               # función base
y = y_true_function + np.random.normal(0, 0.2, len(X))  # función + ruido

# Puntos para predicción suave
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

# Diferentes valores de n_estimators
n_estimators_list = [1, 5, 20, 100]

plt.figure(figsize=(12, 8))

for i, n in enumerate(n_estimators_list, 1):
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X, y)
    y_pred = rf.predict(X_test)

    # Calcular error MSE
    mse = mean_squared_error(y_true, y_pred)

    # Visualización
    plt.subplot(2, 2, i)
    plt.scatter(X, y, s=20, edgecolor="k", c="blue", label="Datos reales")
    plt.plot(X_test, y_true, color="green", linewidth=2, label="Función real")
    plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicción RF")
    plt.title(f"Random Forest (n_estimators={n})\nMSE={mse:.3f}")
    plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


#=======================================================================================
#   Generar un dataset sintético de regresión
X, y = make_regression(n_samples=500, n_features=5, noise=15, random_state=42)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#=======================================================================================
#   Rango de estimadores a probar
estimators_range = [1, 5, 10, 20, 35, 50, 100, 200]

mse_scores = []
r2_scores = []


#=======================================================================================
#   Entrenar un modelo por cada número de estimadores
for n in estimators_range:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)


#=======================================================================================
#   Graficar desempeño
fig, ax1 = plt.subplots(figsize=(8,5))

color = 'tab:blue'
ax1.set_xlabel("Número de estimadores (n_estimators)")
ax1.set_ylabel("Error cuadrático medio (MSE)", color=color)
ax1.plot(estimators_range, mse_scores, marker='o', color=color, label="MSE")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Eje secundario para R^2
color = 'tab:green'
ax2.set_ylabel("Coeficiente de determinación $R^2$", color=color)
ax2.plot(estimators_range, r2_scores, marker='s', linestyle='--', color=color, label="$R^2$")
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Desempeño del Random Forest en función de n_estimators")
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ===========================
# Cargar conjunto de datos
# ===========================
california = fetch_california_housing()
X, y = california.data, california.target

# Para visualización, tomamos solo una característica (ej: MedInc = índice 0)
X_feature = X[:, [0]]  # Median income
feature_name = california.feature_names[0]

# ===========================
# División entrenamiento / prueba
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_feature, y, test_size=0.3, random_state=42
)

# ===========================
# Entrenar modelo
# ===========================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ===========================
# Visualización
# ===========================
# Crear una malla de valores de entrada para graficar predicciones
X_plot = np.linspace(X_feature.min(), X_feature.max(), 500).reshape(-1, 1)
y_pred = rf.predict(X_plot)

plt.figure(figsize=(10,6))

# Datos reales (puntos)
plt.scatter(X_train, y_train, color="lightgray", alpha=0.5, label="Datos de entrenamiento")
plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="Datos de prueba")

# Predicción del modelo (curva)
plt.plot(X_plot, y_pred, color="red", linewidth=2, label="Predicción Random Forest")

plt.title(f"Aproximación con Random Forest (Característica: {feature_name})")
plt.xlabel(feature_name)
plt.ylabel("Valor medio de la vivienda (en $100,000)")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


#=======================================================================================
#   Cargar dataset California Housing
data = fetch_california_housing()
X, y = data.data, data.target


#=======================================================================================
#   Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#=======================================================================================
#   Evaluar desempeño para distintos n_estimators
estimators_range = [1, 5, 10, 20, 35, 50, 100, 200]
mse_scores = []
r2_scores = []

for n in estimators_range:
    rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)


#=======================================================================================
#   Graficar resultados
fig, ax1 = plt.subplots(figsize=(8,5))

color = 'tab:blue'
ax1.set_xlabel("Número de estimadores")
ax1.set_ylabel("MSE", color=color)
ax1.plot(estimators_range, mse_scores, marker='o', color=color, label="MSE")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("R²", color=color)
ax2.plot(estimators_range, r2_scores, marker='s', color=color, label="R²")
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Desempeño del Random Forest en California Housing")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------------
# Generar datos sintéticos
# ---------------------------
np.random.seed(42)
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(scale=0.2, size=X.shape[0])

# ---------------------------
# Definir modelo GBM
# ---------------------------
gbm = GradientBoostingRegressor(
    n_estimators=5,      # número pequeño de iteraciones para visualizar
    learning_rate=0.5,   # peso de cada árbol débil
    max_depth=2,
    random_state=42
)

gbm.fit(X, y)

# ---------------------------
# Graficar cada iteración
# ---------------------------
xx = np.linspace(0, 10, 200)[:, np.newaxis]

plt.figure(figsize=(10, 6))

# Línea base: media inicial
init_pred = np.full_like(xx.ravel(), gbm.init_.constant_, dtype=float)
plt.plot(xx, init_pred, "--", color="black", label="Predicción inicial (constante)")

# Predicciones acumuladas
pred_acum = init_pred.copy()

# Iterar sobre los árboles débiles
for i, stage_pred in enumerate(gbm.staged_predict(xx), 1):
    plt.plot(xx, stage_pred, color="gray", alpha=0.6)  # árboles débiles (gris)
    pred_acum = stage_pred  # la predicción acumulada hasta ese punto

# Línea azul = modelo final
plt.plot(xx, pred_acum, color="blue", linewidth=2, label="Predicción acumulada (final)")

# Datos reales
plt.scatter(X, y, color="red", s=25, label="Datos")

plt.title("Ejemplo Visual de Gradient Boosting")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss


#===================================================================
# Generar dataset sintético
X, y = make_classification(
    n_samples=2000, n_features=20,
    n_informative=15, n_redundant=5,
    random_state=42
)


#===================================================================
#  Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#===================================================================
#  Definir modelo Gradient Boosting
n_estimators = 200
gbm = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


#===================================================================
#  Entrenar el modelo
gbm.fit(X_train, y_train)

# Guardar errores de log-loss en cada iteración
train_loss = []
val_loss = []

for y_pred_train, y_pred_val in zip(
    gbm.staged_predict_proba(X_train),
    gbm.staged_predict_proba(X_val)
):
    train_loss.append(log_loss(y_train, y_pred_train))
    val_loss.append(log_loss(y_val, y_pred_val))

# Graficar evolución de iteraciones
plt.figure(figsize=(10,6))
plt.plot(range(1, n_estimators+1), train_loss, label="Entrenamiento", linewidth=2)
plt.plot(range(1, n_estimators+1), val_loss, label="Validación", linewidth=2, linestyle="--")
plt.xlabel("Número de Iteraciones (Árboles)", fontsize=12)
plt.ylabel("Log-loss", fontsize=12)
plt.title("Evolución del Desempeño en GBM", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


#===================================================================
#  Generar datos no lineales
np.random.seed(42)
X = np.linspace(0, 10, 200).reshape(-1, 1)
y_true = np.sin(X).ravel()  # función real
y = y_true + np.random.normal(scale=0.3, size=X.shape[0])  # con ruido


#===================================================================
#  Definir modelo GBM
gbm = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                max_depth=2, random_state=42)

# Entrenamos modelo paso a paso para poder visualizar
gbm.fit(X, y)

# Espacio para predicciones
x_plot = np.linspace(0, 10, 500).reshape(-1, 1)

# --- Figura 1: primeras iteraciones ---
plt.figure(figsize=(8,6))
# Línea real
plt.plot(x_plot, np.sin(x_plot), "k--", label="Función real (sin)")
# Datos
plt.scatter(X, y, c="gray", s=20, label="Datos con ruido", alpha=0.6)

# Mostrar primeras 5 predicciones acumuladas
for i, y_pred in enumerate(gbm.staged_predict(x_plot)):
    if i in [0, 1, 4]:  # algunas iteraciones representativas
        plt.plot(x_plot, y_pred, lw=1.5, alpha=0.8, label=f"Iter {i+1}")

plt.title("GBM Regresión - Primeras Iteraciones")
plt.legend()
plt.tight_layout()
#plt.savefig("GBM_Reg.png", dpi=300)
plt.show()

# --- Figura 2: evolución con más iteraciones ---
plt.figure(figsize=(8,6))
plt.plot(x_plot, np.sin(x_plot), "k--", label="Función real (sin)")
plt.scatter(X, y, c="gray", s=20, alpha=0.6)

for i, y_pred in enumerate(gbm.staged_predict(x_plot)):
    if i in [9, 19, 49]:  # 10, 20 y 50 iteraciones
        plt.plot(x_plot, y_pred, lw=1.5, alpha=0.8, label=f"Iter {i+1}")

plt.title("GBM Regresión - Evolución con Iteraciones")
plt.legend()
plt.tight_layout()
#plt.savefig("GBM_Reg2.png", dpi=300)
plt.show()

# --- Figura 3: predicción final ---
y_pred_final = gbm.predict(x_plot)

plt.figure(figsize=(8,6))
plt.plot(x_plot, np.sin(x_plot), "k--", lw=2, label="Función real (sin)")
plt.scatter(X, y, c="gray", s=20, alpha=0.6, label="Datos con ruido")
plt.plot(x_plot, y_pred_final, "b", lw=2, label="Predicción final GBM")

plt.title("GBM Regresión - Predicción Final")
plt.legend()
plt.tight_layout()
#plt.savefig("GBM_Reg3.png", dpi=300)
plt.show()

# --- Crear figura con 3 columnas ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Figura 1: primeras iteraciones ---
axes[0].plot(x_plot, np.sin(x_plot), "k--", label="Función real (sin)")
axes[0].scatter(X, y, c="gray", s=20, alpha=0.6, label="Datos con ruido")
for i, y_pred in enumerate(gbm.staged_predict(x_plot)):
    if i in [0, 1, 4]:
        axes[0].plot(x_plot, y_pred, lw=1.5, alpha=0.8, label=f"Iter {i+1}")
axes[0].set_title("Primeras Iteraciones")
axes[0].legend()

# --- Figura 2: evolución con más iteraciones ---
axes[1].plot(x_plot, np.sin(x_plot), "k--", label="Función real (sin)")
axes[1].scatter(X, y, c="gray", s=20, alpha=0.6)
for i, y_pred in enumerate(gbm.staged_predict(x_plot)):
    if i in [9, 29, 49]:
        axes[1].plot(x_plot, y_pred, lw=1.5, alpha=0.8, label=f"Iter {i+1}")
axes[1].set_title("Evolución con Iteraciones")
axes[1].legend()

# --- Figura 3: predicción final ---
y_pred_final = gbm.predict(x_plot)
axes[2].plot(x_plot, np.sin(x_plot), "k--", lw=2, label="Función real (sin)")
axes[2].scatter(X, y, c="gray", s=20, alpha=0.6, label="Datos con ruido")
axes[2].plot(x_plot, y_pred_final, "b", lw=2, label="Predicción final GBM")
axes[2].set_title("Predicción Final")
axes[2].legend()

plt.tight_layout()
plt.show()


