import string
import pybrain as pb
import numpy as np
import pandas as pd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#Importamos excel con datos para el input
basedatos = pd.read_excel(r'./Historico.xlsx', sheet_name='Table 1')
columnas = basedatos.columns
columnas = columnas.tolist()
valores = basedatos.values

#df = pd.DataFrame(lista)
#output = df.to_excel('output.xlsx')

network = buildNetwork(5, 4, 1, bias=True, hiddenclass=TanhLayer)

# Creating a dataset that match with the network input and output sizes
# Creamos un conjunto de datos que matchee con los tamaños de entrada y salida de la red
nand_gate = SupervisedDataSet(5, 1)
  
# Creating a dataset for testing
# Creamos un conjunto de datos para testing
nand_train = SupervisedDataSet(5, 1)

for i in valores:

    # Fit input and target values to dataset parameters for nand_train truth table
    # Ajustamos los valores de entrada y los objetivos a los parámetros del conjunto de datos para la tabla de verdad nand_gain
    nand_gate.addSample((i[1], i[2], i[3], i[4], i[5]), (i[6],))

# Fit input and target values to dataset parameters for nand_train truth table
# Ajustamos los valores de entrada y los objetivos a los parámetros del conjunto de datos para la tabla de verdad nand_train 
for i in valores:
  
  nand_train.addSample((i[1], i[2], i[3], i[4], i[5]), (i[6],))

# Training the network with dataset nand_gate
# Entrenamos la red con los conjuntos de datos nand_gate
trainer = BackpropTrainer(network, nand_gate)
  
# Iterate 10 times to train the network
# Iteramos 100 veces para entrenar la red
for epoch in range(100):
    trainer.train()
    trainer.testOnData(dataset=nand_train, verbose=True)