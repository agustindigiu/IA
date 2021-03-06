import pandas as pd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#Importamos los excel con datos para el input
basedatos = pd.read_excel(r'./Historico.xlsx', sheet_name='Table 1')
columnas = basedatos.columns
columnas = columnas.tolist()
valores = basedatos.values

bd = pd.read_excel(r'./Validacion.xlsx', sheet_name='Table 1')
columns = bd.columns
columns = columns.tolist()
values = bd.values

# Definimos la red neuronal artificial con 6 nodos de entrada, 4 nodos ocultos y 1 nodo de salida
network = buildNetwork(6, 4, 1, bias=True, hiddenclass=TanhLayer)

# Creamos un dataset que matchee con los tamaños de entrada y salida de la red
entrenamiento = SupervisedDataSet(6, 1)
  
# Creamos un dataset para testing
validacion = SupervisedDataSet(6, 1)

for i in valores:

    # Ajustamos el input y los objetivos a los parámetros del dataset para el entrenamiento de la red
  entrenamiento.addSample((i[1], i[2], i[3], i[4], i[6], i[7]), (i[8],))

for i in values:

    # Ajustamos el input y los objetivos a los parámetros del dataset para la validacion de los datos de entrenamiento de la red
  validacion.addSample((i[1], i[2], i[3], i[4], i[6], i[7]), (i[8],))

# Entrenamos la red con el dataset entrenamiento
trainer = BackpropTrainer(network, dataset=entrenamiento, learningrate=0.1, lrdecay=1)

# Iteramos 10 veces para entrenar la red
for epoch in range(10):
  trainer.train()

trainer.testOnData(dataset=validacion, verbose=True)