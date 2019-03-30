#!/usr/bin/python3
import sys
import csv
import _pickle as pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model

# leer nombre del archivo csv (parámetro)
wine_file = sys.argv[1]

#  cargar los clasificadores
with open('classifierDTC_5.pkl', 'rb') as fid:
    tree_loaded = pickle.load(fid)

with open('classifier_LR.pkl', 'rb') as fid:
    linear_model_loaded = pickle.load(fid)

# leer csv
wines = pd.read_csv(wine_file,sep=",")
X = wines.values

# Se realizan las predicciones de la clase (binaria)
predictions_tree = tree_loaded.predict(X[:,[10,1,7,5,2]])

# Se realiza la predicción de la calidad
predictions_linear_model = linear_model_loaded.predict(X)

# se redondean los resultados obtenidos
predictions_linear_model = predictions_linear_model.round()

# Se unen las dos predicciones en un DataFrame
df = pd.DataFrame({'class':predictions_tree,'quality':predictions_linear_model})

# Se cambian las respuestas binarias por texto
df.loc[df["class"] == 1,"class"] = 'BUENO'
df.loc[df["class"] == 0,"class"] = 'MALO'

# Se guarda el archivo separado por ;
df.to_csv('output.csv',',',index=False )

print("finished, the output is at 'output.csv' ")
