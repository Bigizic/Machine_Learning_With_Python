#!/usr/bin/env python3
"""linear regression solution to the rain prediction problem
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
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

def linear_regression():
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
    from a categorical column to a binary column. do not use the get_dummies
    method because we would end up with two columns for 'RainTomorrow'
    and we do not want, since 'RainTomorrow' is our target.
    """

    df_sydney_processed.replace(['No', 'Yes'], [0, 1], inplace=True)

    # PERFROM TRAIN AND TEST SPLIT
    # set our 'features' or x values and our Y or target variable.
    df_sydney_processed.drop('Date', axis=1, inplace=True)

    df_sydney_processed = df_sydney_processed.astype(float)

    # features captures every column except RainTomorrow
    features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
    Y = df_sydney_processed['RainTomorrow']

    # Train and Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(features, Y,
                                                        test_size=0.2,
                                                        random_state=10)
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
    report = {"Metrics": ["MAE", "MSE", "R2 Score"],
              "Result": [LinearRegression_MAE,
                         LinearRegression_MSE,
                         LinearRegression_R2]
              }

    pd.DataFrame(report)

    warnings.warn = warn
