#!/usr/bin/env python3
"""Python script that converts a linear classifier into a softmax
regression model
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


def decision_boundary (X,y,model,iris, two=None):
    """
    This function plots a different decision boundary

    @params:
        - X: X dataset
        - y: y dataset
        - model: model
        - iris: iris
        - two: two
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z,cmap=plt.cm.RdYlBu)

    if two:
        cs = plt.contourf(xx, yy, Z,cmap=plt.cm.RdYlBu)
        for i, color in zip(np.unique(y), plot_colors):

            idx = np.where( y== i)
            plt.scatter(X[idx, 0], X[idx, 1],
                        label=y,cmap=plt.cm.RdYlBu, s=15)
        plt.show()

    else:
        set_={0,1,2}
        print(set_)
        for i, color in zip(range(3), plot_colors):
            idx = np.where( y== i)
            if np.any(idx):

                set_.remove(i)

                plt.scatter(X[idx, 0], X[idx, 1],
                            label=y,cmap=plt.cm.RdYlBu,
                            edgecolor='black', s=15)


        for  i in set_:
            idx = np.where( iris.target== i)
            plt.scatter(X[idx, 0], X[idx, 1], marker='x',color='black')

        plt.show()


def plot_probability_array(X,probability_array):
    """
    This function will plot the probability of belonging to each class
    each column is the probability of belonging to a class and the row
    number is the sample number

    @params:
        - X: X dataset
        - probability_array: an array of probabilities dataset
    """
    plot_array=np.zeros((X.shape[0],30))
    col_start=0
    ones=np.ones((X.shape[0],30))
    for class_,col_end in enumerate([10,20,30]):
        plot_array[:,col_start:col_end] = np.repeat(
                   probability_array[:,class_].reshape(-1,1), 10,axis=1)
        col_start=col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    plt.show()


plot_colors = "ryb"
plot_step = 0.02

"""
we will use the iris dataset, it consists of three different
types of irises’ (Setosa y=0, Versicolour y=1, and Virginica y=2),
petal and sepal length, stored in a 150x4 numpy.ndarray.

The rows being the samples and the columns: Sepal Length, Sepal Width,
Petal Length and Petal Width.

The following plot uses the second two features:
"""

pair=[1, 3]
iris = datasets.load_iris()
X = iris.data[:, pair]
y = iris.target
print(np.unique(y))


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")


# SOFTMAX REGRESSION
"""
SoftMax regression is similar to logistic regression,
and the softmax function converts the actual distances
"""

lr = LogisticRegression(random_state=0).fit(X, y)
probability=lr.predict_proba(X)

# plot probability array
plot_probability_array(X,probability)

# stdout output for probability array
print("probability array: ", probability[0,:], "\n")

# see it sums to one
print("Sum of probability: ", probability[0,:].sum(), "\n")

# apply the argmax function
print("Argmax score: ", np.argmax(probability[0,:]), "\n")

# apply the argmax to each sample
softmax_prediction=np.argmax(probability,axis=1)
print("softmax prediction array: ", softmax_prediction, "\n")

# verify that sklearn does this under the hood by comparing it to
# the output of the method  predict
yhat =lr.predict(X)
print("Accuracy score: ", accuracy_score(yhat,softmax_prediction), "\n")
