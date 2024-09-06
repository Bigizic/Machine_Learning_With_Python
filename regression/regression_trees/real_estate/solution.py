#!/usr/bin/env python3
"""A python script that implement regression trees using ScikitLearn
and determine the tree accuracy
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("real_estate_data.csv")

print(data.head())
print(data.shape)
"""Most of the data is valid, there are rows with missing values
which we will deal with in pre-processing
"""
print(data.isna().sum())

# ~~~~ DATA PRE_PROCESSING ~~~~

# drop the rows with missing values cuz we have enough data in the dataset
data.dropna(inplace=True)

# now we can see the dataset has no missing values
print("\n\n")
print(data.isna().sum())

#split dataset into features and what we're predecting (target)(price of house)
X = data.drop(columns=["MEDV"])
y = data["MEDV"]

print(X.head())
print(y.head())


# split data into a training and testing dataset using train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=1)
# create regression tree
regression_tree = DecisionTreeRegressor(criterion = 'mse')

# Training
regression_tree.fit(X_train, y_train)

# Evaluate the score method is also the R^2 value
print("MSE R^2 value: ", regression_tree.score(X_test, y_test))

prediction = regression_tree.predict(X_test)

# printing the average error
# abs = absolute
print("[MSE] average error: $", (prediction - y_test).abs().mean()*1000)


# the training above is for the MSE (MEAN SQUARED ERROR)

# training below is for the MAE (MEAN ABSOLUTE ERROR)

regression_tree = DecisionTreeRegressor(criterion = 'mae')
regression_tree.fit(X_train, y_train)
print("\n\n")
print("MAE R^2 value: ", regression_tree.score(X_test, y_test))
prediction = regression_tree.predict(X_test)
print("MAE average error: $",(prediction - y_test).abs().mean()*1000)


