#!/usr/bin/env python3
"""Decision tree model
"""

import piplite
await piplite.install(['pandas'])
await piplite.install(['matplotlib'])
await piplite.install(['numpy'])
await piplite.install(['scikit-learn'])

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


path="drug200.csv"

my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])

print(my_data.shape)

# pre-processing
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

# fill the target variable.

y = my_data["Drug"]
y[0:5]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y,
                                               test_size=0.3, random_state=3)

# Modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree)

drugTree.fit(X_trainset,y_trainset)

# prediction
predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset,
                                                           predTree))

# visualization
tree.plot_tree(drugTree)
plt.show()
