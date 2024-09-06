#!/usr/bin/env python3
"""A python script that solves the customer churn problem using
logistic regression
"""

import pandas as pd
from matplotlib import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, jaccard_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')


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


churn_data = pd.read_csv("ChurnData.csv")
print(churn_data.head(), "\n")

processed_churn_data = churn_data[['tenure', 'age', 'address', 'income', 'ed',
                                   'employ', 'equip', 'callcard', 'wireless',
                                   'churn',]]
processed_churn_data['churn'] = processed_churn_data['churn'].astype('int')
print(processed_churn_data.head(), "\n")

# print number or rows and columns in the dataset
print("Number of rows and columns are: ", processed_churn_data.shape, "\n")

# define X and Y

X = np.asarray(processed_churn_data[['tenure', 'age', 'address', 'income',
                                     'ed', 'employ', 'equip', 'callcard',
                                     'wireless',]])
print("5 rows of X: ", X[0:5], "\n")

Y = np.asarray(processed_churn_data['churn'])
print("5 rows of Y: ", Y[0:5], "\n")

# normalize the dataset
x = preprocessing.StandardScaler().fit(X).transform(X)
print("5 rows of a preprocessed X dataset: ", x[0:5], "\n")


# Train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2,
                                   random_state=4)

# print number of rows and columns in each train set
print ("Train set: ", X_train.shape,  Y_train.shape)
print ("Test set: ", X_test.shape,  Y_test.shape)


# CREATE LOGISTIC REGRESSION MODEL USING SCIKIT-LEARN
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, Y_train)
print("LOGISTIC REGRESSION MODEL: ", LR, "\n")

yhat = LR.predict(X_test)
print("YHAT: ", yhat, "\n")

yhat_prob = LR.predict_proba(X_test)
print("YHAT PROBABILITY: ", yhat_prob, "\n")


# JACCARD INDEX FOR ACCURACY, OF OUR MODEL
print("JACCARD SCORE: ", jaccard_score(Y_test, yhat, pos_label=0), "\n")

print("Confusion matrix: ", confusion_matrix(Y_test, yhat, labels=[1,0]), "\n")

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],
                      normalize= False, title='Confusion matrix')

# classification report
print("CLASSIFICATION REPORT: ", classification_report(Y_test, yhat), "\n")

# log loss
print("LOG LOSS: ", log_loss(Y_test, yhat_prob))
