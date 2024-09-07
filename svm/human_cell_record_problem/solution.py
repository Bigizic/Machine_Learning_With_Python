#!/usr/bin/env python3
"""A python script that builds and train a machine learning model using human
cell records, to classify if a cell is either benign or malignant.
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cell_data = pd.read_csv("cell_samples.csv")
print(cell_data.head(), "\n")

"""
The ID field contains the patient identifiers.

The characteristics of the cell samples from each patient are contained
in fields Clump to Mit.

The values are graded from 1 to 10, with 1 being the closest to benign.

The Class field contains the diagnosis, as confirmed by separate
medical procedures, as to whether the samples are benign (value = 2) or
malignant (value = 4).
"""

# Let's look at the distribution of the classes based on Clump
# thickness and Uniformity of cell size
aux = cell_data[cell_data['Class'] == 4][0:50].plot(kind='scatter', x='Clump',
               y='UnifSize', color='DarkBlue', label='malignant')
cell_data[cell_data['Class'] == 2][0:50].plot(kind='scatter', x='Clump',
          y='UnifSize', color='Yellow', label='benign', ax=aux)
# plt.show()

# Data pre-processing and selection
print("Columns data types: ", cell_data.dtypes, "\n")

cell_dff = cell_data[pd.to_numeric(cell_data['BareNuc'],
                                   errors='coerce').notnull()]
cell_dff['BareNuc'] = cell_dff['BareNuc'].astype('int')
print("Cell Features type: ", cell_dff.dtypes, "\n")

feature_df = cell_dff[['Clump', 'UnifSize', 'UnifShape', 'MargAdh',
                       'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl',
                       'Mit']]
# pick x and y
X = np.asarray(feature_df)
print("5 rows of x: ", X[0:5], "\n")

Y = np.asarray(cell_dff['Class'])
print("5 rows of y: ", Y[0:5], "\n")


# Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)


# SVM MODELING usnig Radial Basis Function kernelling model
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train)

yhat = clf.predict(X_test)
print("Y hat: ", yhat[0:5], "\n")


# EVALUATE

# compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(Y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'],
                      normalize=False, title='Confusion_matrix')

# use the f1 score from sklearn library
print("F1 score: ", f1_score(Y_test, yhat, average='weighted'), "\n")

# Jaccard index for accuracy
print("Jaccard index: ", jaccard_score(Y_test, yhat, pos_label=2), "\n")


# Using a 'linear' kernel function for mapping
new_clf = svm.SVC(kernel='linear')
new_clf.fit(X_train, Y_train)
new_yhat = new_clf.predict(X_test)
print("Linear kernel y hat: ", new_yhat [0:5], "\n")
print("Linear kernel F1 score: ", f1_score(Y_test,
       new_yhat, average='weighted'), "\n")
print("Linear kernelJaccard score: ", jaccard_score(Y_test,
       new_yhat, pos_label=2))

