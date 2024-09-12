#!/usr/bin/env python3
"""A python script that creates and train a logistic regression
model to solve the weather prediction problem,
proceeds to import variables from linear regression
"""

from sklearn.linear_model import LogisticRegression as LR
from linear_regression import features, Y
from sklearn.model_selection import train_test_split
from sklearn import metrics


def logistic_regression():
    n_x_train, n_x_test, n_y_train, n_y_test = train_test_split(features,
                                                                Y,
                                                                test_size=0.2,
                                                                random_state=1)

    lr = LR(C=0.01, solver='liblinear').fit(n_x_train, n_y_train)

    predict = lr.predict(n_x_test)

    predict_proba = lr.predict_proba(n_x_test)

    LR_Accuracy_Score = metrics.accuracy_score(n_y_test, predict)
    LR_JaccardIndex = metrics.jaccard_score(n_y_test, predict)
    LR_F1_Score = metrics.f1_score(n_y_test, predict)
    LR_Log_Loss = metrics.log_loss(n_y_test, predict_proba)

    print("\n\n\n LOGISTIC REGRESSION \n")
    print("LR Accuracy Score: ", LR_Accuracy_Score)
    print("LR Jaccard Index: ", LR_JaccardIndex)
    print("LR F1 Score: ", LR_F1_Score)
    print("LR Log Loss: ", LR_Log_Loss)

