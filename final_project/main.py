#!/usr/bin/env python3
"""Entry point for final project
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import svm
from sklearn.metrics import jaccard_score, r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import warnings


# Surpress warnings:
def warn(*args, **kwargs):
    pass
warnings.warn = warn


class Classification():

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.Y = None
        self.features = None

    def linear_regression(self):
        # read csv file
        df = pd.read_csv("Weather_Data.csv")
        # print table
        print(df.head())

        """
        process data and perform one hot encoding to convert categorical
        variables to binary variables. Categorical variables like Yes/No
        would be converted to either 0 or 1
        """
        df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday',
                                                               'WindGustDir',
                                                               'WindDir9am',
                                                               'WindDir3pm'])
        """
        Next, replace the values of the 'RainTomorrow' column changing them
        from a categorical column to a binary column. do not use the get
        dummies method because we would end up with two columns for
        'RainTomorrow' and we do not want, since 'RainTomorrow' is our target.
        """

        df_sydney_processed.replace(['No', 'Yes'], [0, 1], inplace=True)

        # PERFROM TRAIN AND TEST SPLIT
        # set our 'features' or x values and our Y or target variable.
        df_sydney_processed.drop('Date', axis=1, inplace=True)

        df_sydney_processed = df_sydney_processed.astype(float)

        # features captures every column except RainTomorrow
        features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
        self.features = features
        Y = df_sydney_processed['RainTomorrow']
        self.Y = Y

        # Train and Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(features, Y,
                                                            test_size=0.2,
                                                            random_state=10)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        # Linear Regression Model Training
        LinearReg = LinearRegression()
        LinearReg.fit(X_train, Y_train)

        predictions = LinearReg.predict(X_test)

        # Linear Mean squared error, mean absolute error and R2 score
        LinearRegression_MAE = np.mean(np.absolute(predictions - Y_test))
        LinearRegression_MSE = np.mean((predictions - Y_test) ** 2)
        LinearRegression_R2 = r2_score(Y_test, predictions)

        print("MAE %.2f: "% LinearRegression_MAE)
        print("MSE %.2f: "% LinearRegression_MSE)
        print("R2 SCORE %.2f: "% LinearRegression_R2)

        # Show the MAE, MSE, and R2 in a tabular format using data frame
        # for the linear model
        report = {"MAE": LinearRegression_MAE,
                  "MSE": LinearRegression_MSE,
                  "R2 SCORE": LinearRegression_R2,
                 }

        print(pd.DataFrame(data=report, index=[0, 1, 2]))

        """L_Accuracy_Score = metrics.accuracy_score(Y_test, predictions)
        L_JaccardIndex = metrics.jaccard_score(Y_test, predictions)
        L_F1_Score = metrics.f1_score(Y_test, predictions)

        print("\n\n\n LINEAR REGRESSION SECTION \n")

        print("LINEAR Accuracy Score: ", L_Accuracy_Score)
        print("LINEAR Jaccard Index: ", L_JaccardIndex)
        print("LINEAR F1 Score: ", L_F1_Score)

        return [L_Accuracy_Score, L_JaccardIndex, L_F1_Score]"""

    def decision_tree(self):
        Tree = DTC(criterion="entropy", max_depth=4).fit(self.X_train,
                                                         self.Y_train)

        predictions = Tree.predict(self.X_test)

        Tree_Accuracy_Score = metrics.accuracy_score(self.Y_test, predictions)
        Tree_JaccardIndex = metrics.jaccard_score(self.Y_test, predictions)
        Tree_F1_Score = metrics.f1_score(self.Y_test, predictions)

        print("\n\n\n DECISION TREE SECTION \n")

        print("DECISION TREE Accuracy Score: ", Tree_Accuracy_Score)
        print("DECISION TREE Jaccard Index: ", Tree_JaccardIndex)
        print("DECISION TREE F1 Score: ", Tree_F1_Score)

        return [Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score]

    def knn(self):
        n = 4
        KNN = KNeighborsClassifier(n_neighbors=n).fit(self.X_train,
                                                      self.Y_train)
        print("\n\n\nTESTING KNN \n")

        predictions = KNN.predict(self.X_test)
        print(predictions[0:5])

        KNN_Accuracy_Score = metrics.accuracy_score(self.Y_test, predictions)
        KNN_JaccardIndex = metrics.jaccard_score(self.Y_test, predictions)
        KNN_F1_Score = metrics.f1_score(self.Y_test, predictions)

        print("KNN Accuracy Score: ", KNN_Accuracy_Score)
        print("KNN Jaccard Index: ", KNN_JaccardIndex)
        print("KNN F1 Score: ", KNN_F1_Score)

        return [KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score]

    def logistic_regression(self):
        x_train, x_test, y_train, y_test = train_test_split(self.features,
                                                           self.Y,
                                                           test_size=0.2,
                                                           random_state=1)

        lr = LR(C=0.01, solver='liblinear').fit(x_train, y_train)

        predict = lr.predict(x_test)

        predict_proba = lr.predict_proba(x_test)

        LR_Accuracy_Score = metrics.accuracy_score(y_test, predict)
        LR_JaccardIndex = metrics.jaccard_score(y_test, predict)
        LR_F1_Score = metrics.f1_score(y_test, predict)
        LR_Log_Loss = metrics.log_loss(y_test, predict_proba)

        print("\n\n\n LOGISTIC REGRESSION \n")
        print("LR Accuracy Score: ", LR_Accuracy_Score)
        print("LR Jaccard Index: ", LR_JaccardIndex)
        print("LR F1 Score: ", LR_F1_Score)
        print("LR Log Loss: ", LR_Log_Loss)

        return [LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score, LR_Log_Loss]

    def svm(self):
        SVM = svm.SVC(kernel='rbf').fit(self.X_train, self.Y_train)

        predictions = SVM.predict(self.X_test)

        SVM_Accuracy_Score = metrics.accuracy_score(self.Y_test, predictions)
        SVM_JaccardIndex = metrics.jaccard_score(self.Y_test, predictions)
        SVM_F1_Score = metrics.f1_score(self.Y_test, predictions)

        print("\n\n\n SVM MODEL \n")
        print("SVM Accuracy Score: ", SVM_Accuracy_Score)
        print("SVM Jaccard Index: ", SVM_JaccardIndex)
        print("SVM F1 Score: ", SVM_F1_Score)

        return [SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score]

if __name__ == '__main__':
    clf = Classification()
    clf.linear_regression()
    knn = clf.knn()
    tree = clf.decision_tree()
    logistic_regression = clf.logistic_regression()
    svm = clf.svm()
    data = {"MODEL": ["KNN",
                      "Decision Tree",
                      "Logistic Regression",
                      "SVM"],
            "ACCURACY SCORE": [knn[0],
                               tree[0],
                               logistic_regression[0],
                               svm[0],
                               ],
            "JACCARD INDEX": [knn[1],
                              tree[1],
                              logistic_regression[1],
                              svm[1],],
            "F1-SCORE": [knn[2],
                         tree[2],
                         logistic_regression[2],
                         svm[2],],
            "LOGLOSS": [None, None, logistic_regression[3], None],
           }
    print(pd.DataFrame(data=data))
