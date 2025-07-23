# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1JyuFdMP2jhnTYobc4YRb2eCwLnEsADrR
"""

import tensorflow as tf
import numpy as np

celsius = np.array([-40,-10,0,8,15,22,38,50,100], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100,122,212], dtype=float)

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=[1], activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("hagamos una prediccion!")
entrada = np.array([[100.0]], dtype=np.float32)
resultado = modelo.predict(entrada, verbose=0)
print("El resultado es "+str(resultado) + "fahrenheit")

print("variable internas del modelo")
print(capa.get_weights())

modelo.summary()