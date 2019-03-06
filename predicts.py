#!/usr/bin/python3
import sys
import csv
import _pickle as pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model

# save the classifier
with open('classifier.pkl', 'rb') as fid:
    tree_loaded = pickle.load(fid)

with open('classifier_LR.pkl', 'rb') as fid:
    linear_model_loaded = pickle.load(fid)

wine_file = sys.argv[1]

wines = pd.read_csv(wine_file,sep=";")
X = wines.drop(['quality'], axis=1).values

predictions = tree_loaded.predict(X[:,[10,1,7,5,2]])
predictions_linear_model = linear_model_loaded.predict(X)

with open(wine_file,'r') as csvinput:
    with open('output.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('Result')
        all.append(row)

        for idx, row in enumerate(reader):
            row.append(predictions[idx])
            all.append(row)

        writer.writerows(all)
