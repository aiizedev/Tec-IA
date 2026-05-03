from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Cargar el dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target

# 2. División estratificada (60% Train, 20% Val, 20% Test)
semilla = 42
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=semilla, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=semilla, stratify=y_temp
)

# 3. Verificación
print(f"Entrenamiento: {len(X_train)} | Validación: {len(X_val)} | Prueba: {len(X_test)}")

import matplotlib.pyplot as plt

def visualizar_distribucion_clase(id_persona=0):
    """Muestra las 10 fotos de una persona y en qué set quedaron"""
    # Encontrar los índices en cada set para esa persona
    idx_train = np.where(y_train == id_persona)[0]
    idx_val = np.where(y_val == id_persona)[0]
    idx_test = np.where(y_test == id_persona)[0]
    
    # Recopilar imágenes y sus etiquetas de origen
    imagenes = [X_train[i] for i in idx_train] + [X_val[i] for i in idx_val] + [X_test[i] for i in idx_test]
    labels = ["Entrenamiento"]*len(idx_train) + ["Validación"]*len(idx_val) + ["Prueba"]*len(idx_test)
    
    # Crear el grid de 2x5 (total 10 imágenes por persona)
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Distribución de imágenes para la Persona {id_persona}", fontsize=16)
    
    for i in range(len(imagenes)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imagenes[i].reshape(64, 64), cmap='gray')
        color_label = 'blue' if labels[i] == "Entrenamiento" else ('orange' if labels[i] == "Validación" else 'green')
        plt.title(labels[i], color=color_label)

        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Visualizamos la Persona 0 para comprobar la estratificación (6 Train, 2 Val, 2 Test)
visualizar_distribucion_clase(id_persona=11)

from sklearn.preprocessing import StandardScaler

# 1. Instanciar el escalador para Normalización Z-score
# Esto restará la media y dividirá por la desviación estándar: (x - u) / s
scaler = StandardScaler()

# 2. Ajustar el escalador ÚNICAMENTE con el conjunto de entrenamiento (fit)
# Es CRUCIAL no incluir validación ni prueba aquí para evitar que el modelo 
# "conozca" la distribución de los datos que usaremos para evaluar.
scaler.fit(X_train)

# 3. Transformar los tres conjuntos de forma coherente
# Usamos la media y desviación calculadas solo con el set de entrenamiento
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Verificación técnica de la normalización
print("Validación de la Normalización Z-score:")
print(f"-> Entrenamiento - Media: {X_train_scaled.mean():.2f} | Desv. Estándar: {X_train_scaled.std():.2f}")
print(f"-> Validación    - Media: {X_val_scaled.mean():.2f} | Desv. Estándar: {X_val_scaled.std():.2f}")
print(f"-> Prueba        - Media: {X_test_scaled.mean():.2f} | Desv. Estándar: {X_test_scaled.std():.2f}")

# A partir de ahora, usaremos las variables '_scaled' para alimentar la red neuronal (MLP)

import numpy as np
import time
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.initializers import HeNormal, LecunNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Fijamos semillas para que los números "aleatorios" sean idénticos si lo vuelves a correr
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def build_model(profile, lr, dropout_rate=None, l2_reg=None, neurons_list=[]):
    model = Sequential()
    model.add(Input(shape=(4096,)))
    
    if profile == 'Robusto':
        # Perfil Robusto: ReLU, HeNormal, Dropout, Adam
        for units in neurons_list:
            model.add(Dense(units, activation='relu', kernel_initializer=HeNormal()))
            model.add(Dropout(dropout_rate))
        optimizer = Adam(learning_rate=lr)
        
    elif profile == 'Profundo':
        # Perfil Profundo: SELU, LecunNormal, L2 Weight Decay, AdamW
        for units in neurons_list:
            model.add(Dense(units, activation='selu', kernel_initializer=LecunNormal(), kernel_regularizer=l2(l2_reg)))
        
        # Uso de AdamW (Fallback a Adam si la versión de TF es muy antigua)
        try:
            from tensorflow.keras.optimizers import AdamW
            optimizer = AdamW(learning_rate=lr, weight_decay=l2_reg)
        except ImportError:
            optimizer = Adam(learning_rate=lr) 
                
    elif profile == 'Adaptativo':
        # Perfil Adaptativo: LeakyReLU, HeNormal, RMSprop
        for units in neurons_list:
            model.add(Dense(units, kernel_initializer=HeNormal()))
            model.add(LeakyReLU(alpha=0.1))
        optimizer = RMSprop(learning_rate=lr)
        
    # Capa de salida común: 40 clases, Softmax
    model.add(Dense(40, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

NUM_ITERACIONES = 50
PACIENCIA = 12

resultados = []
print(f"Iniciando Búsqueda Aleatoria ({NUM_ITERACIONES} iteraciones)...\n")

for i in range(NUM_ITERACIONES):
    profile = random.choice(['Robusto', 'Profundo', 'Adaptativo'])
    lr = 10 ** np.random.uniform(-5, -2) # Tasa logarítmica
    
    # Callback base
    callbacks = [EarlyStopping(monitor='val_loss', patience=PACIENCIA, restore_best_weights=True)]
    kwargs = {}
    
    # Muestreo de hiperparámetros según el perfil
    if profile == 'Robusto':
        kwargs['dropout_rate'] = np.random.uniform(0.3, 0.5)
        n1 = random.choice([256, 512, 1024])
        n2 = random.choice([128, 256]) # Segunda capa
        kwargs['neurons_list'] = [n1, n2]
        
    elif profile == 'Profundo':
        kwargs['l2_reg'] = 10 ** np.random.uniform(-6, -3)
        num_layers = random.choice([3, 4])
        kwargs['neurons_list'] = [512, 256, 128, 64][:num_layers]
        
    elif profile == 'Adaptativo':
        num_layers = random.choice([2, 3])
        kwargs['neurons_list'] = [512, 256, 128][:num_layers]
        # Scheduler exclusivo para Adaptativo
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6))

    # Entrenar
    model = build_model(profile, lr, **kwargs)
    start_time = time.time()
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=150, batch_size=32,
        validation_data=(X_val_scaled, y_val),
        callbacks=callbacks, verbose=0 # Verbose 0 para salida limpia
    )
    tiempo_train = time.time() - start_time
    
    # Métricas del mejor momento (guardado por restore_best_weights)
    val_acc_best = max(history.history['val_accuracy'])
    val_loss_best = min(history.history['val_loss'])
    
    resultados.append({
        'Iteración': i + 1, 'Perfil': profile, 'LR': lr,
        'Hiperparámetros': str(kwargs), 'Val_Accuracy': val_acc_best,
        'Val_Loss': val_loss_best, 'Tiempo (s)': tiempo_train,
        'Historia': history.history, 'Pesos': model.get_weights(), 'kwargs': kwargs
    })
    
    print(f"Iter {i+1:02d} | Perfil: {profile:10s} | Val Acc: {val_acc_best:.4f} | Tiempo: {tiempo_train:.1f}s")

# Selección del Top 3
df_resultados = pd.DataFrame(resultados).drop(columns=['Historia', 'Pesos', 'kwargs'])
df_top3 = df_resultados.sort_values(by='Val_Accuracy', ascending=False).head(3).reset_index(drop=True)

print("\n🏆 --- TABLA COMPARATIVA: TOP 3 MODELOS --- 🏆")
display(df_top3)

# Preparar figuras
fig_curves, axes_curves = plt.subplots(3, 2, figsize=(15, 12))
fig_curves.suptitle('Curvas de Aprendizaje - Entrenamiento vs Validación', fontsize=16)

fig_cm, axes_cm = plt.subplots(1, 3, figsize=(20, 6))
fig_cm.suptitle('Matrices de Confusión en conjunto de PRUEBA', fontsize=16)

for idx, row in df_top3.iterrows():
    datos_modelo = next(item for item in resultados if item["Iteración"] == row["Iteración"])
    
    # Reconstruir modelo ganador
    modelo_ganador = build_model(datos_modelo['Perfil'], datos_modelo['LR'], **datos_modelo['kwargs'])
    modelo_ganador.set_weights(datos_modelo['Pesos'])
    
    hist = datos_modelo['Historia']
    
    # Graficar Loss
    axes_curves[idx, 0].plot(hist['loss'], label='Train Loss')
    axes_curves[idx, 0].plot(hist['val_loss'], label='Val Loss')
    axes_curves[idx, 0].set_title(f"Rank {idx+1}: {row['Perfil']} - Loss")
    axes_curves[idx, 0].legend()
    
    # Graficar Accuracy
    axes_curves[idx, 1].plot(hist['accuracy'], label='Train Acc')
    axes_curves[idx, 1].plot(hist['val_accuracy'], label='Val Acc')
    axes_curves[idx, 1].set_title(f"Rank {idx+1}: {row['Perfil']} - Accuracy")
    axes_curves[idx, 1].legend()
    
    # EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST)
    y_pred_probs = modelo_ganador.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"🥇 RANK {idx+1} | Perfil: {row['Perfil']} | TEST Accuracy: {test_acc:.4f}")
    print(f"{'='*60}")
    
    # Reporte F1, Precision, Recall
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Matriz de Confusión Heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues', ax=axes_cm[idx], cbar=False)
    axes_cm[idx].set_title(f"Rank {idx+1} - {row['Perfil']}\nTest Acc: {test_acc:.2f}")
    axes_cm[idx].set_xlabel('Predicción')
    axes_cm[idx].set_ylabel('Real')

plt.tight_layout()
plt.show() 

import numpy as np
import matplotlib.pyplot as plt

def visualizar_fotos_predichas(modelo, X_imagenes, X_escaladas, y_reales, cantidad=10):
    """
    Toma el modelo, hace predicciones sobre las X_escaladas, 
    pero grafica las X_imagenes (originales) para que se vean bien.
    """
    # 1. Hacer que la red haga sus predicciones
    probabilidades = modelo.predict(X_escaladas, verbose=0)
    predicciones = np.argmax(probabilidades, axis=1)
    
    # 2. Escoger 'cantidad' de índices al azar
    indices = np.random.choice(len(y_reales), cantidad, replace=False)
    
    # 3. Dibujar la cuadrícula
    plt.figure(figsize=(15, 6))
    plt.suptitle("Test de Reconocimiento: ¿Quién es quién?", fontsize=16)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        
        # Recuperar la imagen 64x64 de los datos sin escalar
        foto = X_imagenes[idx].reshape(64, 64)
        plt.imshow(foto, cmap='gray')
        plt.axis('off')
        
        pred = predicciones[idx]
        real = y_reales[idx]
        
        # Color verde si el modelo acertó, rojo si falló
        color = 'green' if pred == real else 'red'
        plt.title(f"El modelo dijo: {pred}\nEra: {real}", color=color, fontweight='bold')
        
    plt.tight_layout()
    plt.show()

# ==========================================
# EJECUCIÓN CON EL MEJOR MODELO (RANK 1)
# ==========================================

# 1. Recuperamos el modelo Rank 1 desde nuestra tabla df_top3
mejor_iteracion = df_top3.iloc[0]['Iteración']
datos_mejor = next(item for item in resultados if item["Iteración"] == mejor_iteracion)

# 2. Lo reconstruimos con sus parámetros ganadores
mejor_modelo_mlp = build_model(datos_mejor['Perfil'], datos_mejor['LR'], **datos_mejor['kwargs'])
mejor_modelo_mlp.set_weights(datos_mejor['Pesos'])

# 3. Llamamos a la función usando X_test (datos originales) y X_test_scaled (para predecir)
visualizar_fotos_predichas(mejor_modelo_mlp, X_test, X_test_scaled, y_test, cantidad=10)

def visualizar_errores_red(modelo, X_imagenes, X_escaladas, y_reales):
    # 1. Hacer predicciones sobre todo el set de prueba
    probabilidades = modelo.predict(X_escaladas, verbose=0)
    predicciones = np.argmax(probabilidades, axis=1)
    
    # 2. Encontrar los índices donde el modelo se equivocó
    indices_errores = np.where(predicciones != y_reales)[0]
    
    if len(indices_errores) == 0:
        print("✨ ¡Perfecto! El modelo no cometió ningún error en el conjunto de prueba.")
        return

    print(f"Se encontraron {len(indices_errores)} errores en total.")
    
    # 3. Mostrar los errores (limitamos a 20 por si hay demasiados)
    num_mostrar = min(len(indices_errores), 20)
    columnas = 5
    filas = (num_mostrar // columnas) + (1 if num_mostrar % columnas != 0 else 0)
    
    plt.figure(figsize=(15, 4 * filas))
    plt.suptitle("Análisis de Errores (Predicción Incorrecta)", fontsize=16, color='red')
    
    for i in range(num_mostrar):
        idx = indices_errores[i]
        plt.subplot(filas, columnas, i + 1)
        
        foto = X_imagenes[idx].reshape(64, 64)
        plt.imshow(foto, cmap='gray')
        plt.axis('off')
        
        plt.title(f"El modelo dijo: {predicciones[idx]}\nEn realidad era: {y_reales[idx]}", 
                  color='darkred', fontsize=10)
        
    plt.tight_layout()
    plt.show()

# Ejecutar el análisis de errores con tu mejor modelo
visualizar_errores_red(mejor_modelo_mlp, X_test, X_test_scaled, y_test)
