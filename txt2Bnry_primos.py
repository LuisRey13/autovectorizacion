import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Obtener los primeros 30 números primos
def primos(n):
  primos = []
  for i in range(2, n + 1):
    es_primo = True
    for j in range(2, int(i**0.5) + 1):
      if i % j == 0:
        es_primo = False
        break
    if es_primo:
      primos.append(i)
  return primos

primos_lista = primos(100)

# Crear los datos de entrenamiento
datos_entrenamiento = []
for i in range(len(primos_lista) - 3):
  datos_entrenamiento.append([primos_lista[i:i+3], primos_lista[i+3]])

# Convertir los datos a arreglos NumPy
X = np.array([x[0] for x in datos_entrenamiento])
y = np.array([x[1] for x in datos_entrenamiento])

# Reshape para alimentar la LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], 1)

# Crear el modelo de secuencia a secuencia
model = keras.Sequential([
  layers.LSTM(300, input_shape=(3, 1),return_sequences=True),  # La primera capa no necesita `return_sequences`
  layers.Dense(300, activation='relu'),
  layers.Dense(30, activation='relu'),
  layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X, y, epochs=10000)

# Evaluar el modelo con ejemplos
ejemplos = [[2, 3, 5], [3, 5, 7],[5, 7, 11],[7,11,13], [13, 17, 19], [79,83,89]]
for ejemplo in ejemplos:
  prediccion = model.predict(np.array([ejemplo]).reshape(1, 3, 1))
  print("Para ",ejemplo," se predice: ",prediccion[0][0])