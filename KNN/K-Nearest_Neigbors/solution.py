#!/usr/bin/env python3
"""KNN Model
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv("teleCust1000t.csv")
print(df.head())

# Data Visualization and Analysis
# Let’s see how many of each class is in our data set

print(df['custcat'].value_counts())

# define feature sets X
print(df.columns)

X = df[['region', 'tenure','age', 'marital',
        'address', 'income', 'ed', 'employ',
        'retire', 'gender', 'reside']] .values  #.astype(float)

print(X[0:5])

# labels
y = df['custcat'].values
y[0:5]

# normalize processing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

# Train test split
X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                     test_size=0.2,
                                                     random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# KNN MODEL

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

# predicting
yhat = neigh.predict(X_test)
yhat[0:5]

# Accuracy evaluation
print("Train set Accuracy: ", metrics.accuracy_score(y_train,
                                                     neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Other K

# calculate the accuracy of KNN for different values of k.

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

# Plot the model accuracy for a different number of neighbors.

plt.plot(range(1, Ks),mean_acc,'g')
plt.fill_between(range(1, Ks),
                 mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1, Ks),
                 mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,
                 color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(),
      "with k=", mean_acc.argmax()+1)
