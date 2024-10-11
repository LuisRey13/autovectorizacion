import os
os.system('cls' if os.name == 'nt' else 'clear')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

Atributos=["Vuela","Camina","Nada","Grande","Mediano","Pequeño",
           "Carnivoro","Hervivoro","aletas","alas","bipedo",
           "cuadrupedo","pelo","plumas","piel","escamas","ovíparos",
           "cola","garras","branqueas"]
Animales=["Buitre", "buho","gallina","pajaro", "perro", "gato", 
          "leon","caballo","vaca","ratón", "delfin","tiburon",
          "pinguino","tortuga","rana","pez"]
AnimalsO=np.array(  [[1,0,0,0,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0], # Buitre
                    [1,0,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,1,0], # Buho
                    [0,1,0,0,0,1,0,1,0,1,1,0,0,1,0,0,1,0,0,0], # Gallina
                    [1,0,0,0,0,1,0,1,0,1,1,0,0,1,0,0,1,0,0,0], # Pajaro
                    [0,1,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0], # Perro
                    [0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0], # Gato
                    [0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,1,1,0], # Leon
                    [0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0], # Caballo
                    [0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0], # Vaca
                    [0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,0,0], # Ratón
                    [0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0], # Delfín
                    [0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1], # Tiburon            
                    [0,1,1,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,0], # Pinguino
                    [0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0], # Tortuga
                    [0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,1,0,0,1], # Rana
                    [0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,0,0,1]])#Pez
NumAnmls=len(AnimalsO)
NumAtrbts=len(Atributos)
vctorIn=2; NeurnIN=60;
AnimalsI=np.random.rand(NumAnmls,vctorIn)

AnimalsI0=np.round(np.array(AnimalsI),6) # para ver en donde iniciaron

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(vctorIn,), name='input_layer'),  # Ahora espera (None, 3)
    tf.keras.layers.Dense(NeurnIN, activation='tanh', name='hidden_layer_1'),
    tf.keras.layers.Dense(30, activation='tanh', name='hidden_layer_2'),
    tf.keras.layers.Dense(NumAtrbts,activation='softmax', name='output_layer')
])
# Pesos iniciales de primer capa
il = model.get_layer(name='hidden_layer_1')
ilw0= [w.numpy() for w in il.weights][0]
# Compilar el modelo
#model.compile(loss='categorical_crossentropy', optimizer='adam')
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.333),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Entrenar el modelo iteración por iteración
loss_history = []  # Lista para guardar el historial de pérdida
AnimalsI_history=[]
epocas = 300;
for e in range(epocas):  # Ajusta el número de epocas según sea necesario
    for i in range(NumAnmls):  # Itera sobre cada muestra de entrenamiento
        # Realiza una actualización de peso para la muestra actual
        model.train_on_batch(AnimalsI[i:i+1], AnimalsO[i:i+1])
        # Obtener los pesos de la capa de entrada por nombre
        ilw = model.get_layer(name='hidden_layer_1').weights
        ilw1= [w.numpy() for w in il.weights][0]
        # Ajuste de entradas:
        NwVIn=[]
        for v in range(vctorIn):
            XW=0
            for w in range(NeurnIN):
                XW+=ilw0[v][w]*(ilw1[v][w]-ilw0[v][w])/AnimalsI[i][v]
            NwVIn.append(AnimalsI[i][v]+XW)
        AnimalsI[i]=NwVIn
    ilw0=ilw1

    loss = model.evaluate(AnimalsI, AnimalsO, verbose=0)
    loss_history.append(loss)  # Añade la pérdida al historial

    AnimalsI_history.append([np.array(AnimalsI[:][Ai][:]) for Ai in range(3)])
    print("Entrenamiento: ",round(e/epocas*100,2),"%, loss:",loss)

print("AnimalsI_history: ",AnimalsI_history)
AnimalsI1=np.round(np.array(AnimalsI),6) # Para ver donde terminaron

predicciones = model.predict(AnimalsI[0:1])
print("predicciones: ")
print("    ")
print(np.round(predicciones,3))
print(AnimalsO[0:1])




#fig = plt.figure()
fig, axes = plt.subplots(2, 2, figsize=(12, 6))  # Crea una figura con 2 subplots
# Gráfico 1: AnimalsI1
ax0 = axes[0,0]
#ax0 = fig.add_subplot(221)
ax0.plot(loss_history)
ax0.set_xlabel("Épocas")
ax0.set_ylabel("Pérdida")
ax0.set_title("Historial de Pérdida")

# Gráfico 2: AnimalsI2
ax1 = axes[0,1]
#ax1 = fig.add_subplot(222)
for i in range(3):
    vctr0=[];vctr1=[]
    for Ai in AnimalsI_history:
        vctr0.append(Ai[i][0])
        vctr1.append(Ai[i][1])
    ax1.plot(vctr0,'-*')
    ax1.plot(vctr1,'-*')
ax1.set_xlabel("Épocas")
ax1.set_ylabel("Evolución")
ax1.set_title("Evolución de la vectorización")


# Gráfico 3: AnimalsI2
ax2 = axes[1,0]
#ax2 = fig.add_subplot(223, projection='3d')
for t in range(len(AnimalsI0)):
    ax2.scatter(AnimalsI0[t][0], AnimalsI0[t][1])
    ax2.text(AnimalsI0[t][0] + (random.random()-0.5)*.1, AnimalsI0[t][1] + (random.random()-0.5)*.1, Animales[t])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('AnimalsI0')

# Gráfico 4: AnimalsI1
ax3 = axes[1,1]
#ax3 = fig.add_subplot(224, projection='3d')
for t in range(len(AnimalsI1)):
    ax3.scatter(AnimalsI1[t][0], AnimalsI1[t][1])
    ax3.text(AnimalsI1[t][0] + (random.random()-0.5)*.1, AnimalsI1[t][1] + (random.random()-0.5)*.1, Animales[t])
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('AnimalsI1')

plt.tight_layout()  # Ajusta el espacio entre los subplots
plt.show()

# Evaluar el modelo
#loss = model.evaluate(AnimalsI,AnimalsO, verbose=0)
#print('Pérdida:', loss)


