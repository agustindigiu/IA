import pybrain as pb
import numpy as np
import pandas as pd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#Importamos el excel con datos para el input
basedatos = pd.read_excel(r'./Historico.xlsx', sheet_name='Table 1')
columnas = basedatos.columns
columnas = columnas.tolist()
valores = basedatos.values

network = buildNetwork(6, 4, 1, bias=True, hiddenclass=TanhLayer)

# Creamos un dataset que matchee con los tamaños de entrada y salida de la red
nand_gate = SupervisedDataSet(6, 1)
  
# Creamos un dataset para testing
nand_train = SupervisedDataSet(6, 1)

for i in valores:

    # Ajustamos el input y los objetivos a los parámetros del dataset para la tabla de verdad nand_gain
    nand_gate.addSample((i[1], i[2], i[3], i[4], i[5], i[6]), (i[7],))
 
for i in valores:

  # Ajustamos el input y los objetivos a los parámetros del dataset para la tabla de verdad nand_train
  nand_train.addSample((i[1], i[2], i[3], i[4], i[5], i[6]), (i[7],))

# Entrenamos la red con el dataset nand_gate
trainer = BackpropTrainer(network, nand_gate)
  
# Iteramos 100 veces para entrenar la red
for epoch in range(5):
    trainer.train()
    trainer.testOnData(dataset=nand_train, verbose=True)