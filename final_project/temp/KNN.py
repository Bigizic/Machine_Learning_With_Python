#!/usr/bin/env python3
"""A python script that creates and train a K-Nearest Neighbor (KNN)
model to solve the weather prediction problem,
proceeds to import variables from linear regression
"""

from linear_regression import X_train, Y_train, X_test, Y_test
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def knn():
    n = 4
    KNN = KNeighborsClassifier(n_neighbors=n).fit(X_train, Y_train)
    print("\n\n\nTESTING KNN \n")

    predictions = KNN.predict(X_test)
    print(predictions[0:5])

    KNN_Accuracy_Score = metrics.accuracy_score(Y_test, predictions)
    KNN_JaccardIndex = metrics.jaccard_score(Y_test, predictions)
    KNN_F1_Score = metrics.f1_score(Y_test, predictions)

    print("KNN Accuracy Score: ", KNN_Accuracy_Score)
    print("KNN Jaccard Index: ", KNN_JaccardIndex)
    print("KNN F1 Score: ", KNN_F1_Score)
