#!/usr/bin/env python3
"""A python script that creates and train a Decision Tree
model to solve the weather prediction problem,
proceeds to import variables from linear regression
"""

from linear_regression import X_train, Y_train, X_test, Y_test
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import metrics


def decision_tree():
    Tree = DTC(criterion="entropy", max_depth=4).fit(X_train, Y_train)

    predictions = Tree.predict(X_test)

    Tree_Accuracy_Score = metrics.accuracy_score(Y_test, predictions)
    Tree_JaccardIndex = metrics.jaccard_score(Y_test, predictions)
    Tree_F1_Score = metrics.f1_score(Y_test, predictions)

    print("\n\n\n DECISION TREE SECTION \n")

    print("KNN Accuracy Score: ", Tree_Accuracy_Score)
    print("KNN Jaccard Index: ", Tree_JaccardIndex)
    print("KNN F1 Score: ", Tree_F1_Score)
