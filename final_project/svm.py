#!/usr/bin/env python3
"""A python script that creates and train a SUPPORT VECTOR MACHINE(SVM)
model to solve the weather prediction problem,
proceeds to import variables from linear regression
"""

from linear_regression import X_train, Y_train, X_test, Y_test
from sklearn import metrics
from sklearn import svm


def svm():
    SVM = svm.SVC(kernel='rbf').fit(X_train, Y_train)

    predictions = SVM.predict(X_test)

    SVM_Accuracy_Score = metrics.accuracy_score(Y_test, predictions)
    SVM_JaccardIndex = metrics.jaccard_score(Y_test, predictions)
    SVM_F1_Score = metrics.f1_score(Y_test, predictions)

    print("\n\n\n SVM MODEL \n")
    print("SVM Accuracy Score: ", SVM_Accuracy_Score)
    print("SVM Jaccard Index: ", SVM_JaccardIndex)
    print("SVM F1 Score: ", SVM_F1_Score)
