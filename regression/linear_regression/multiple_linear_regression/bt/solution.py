#!/usr/bin/env python3
""" Simple regression script
"""

import glob
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

csv_files = glob.glob("csv/*.csv")
df_list = [pd.read_csv(filee) for filee in csv_files]
df = pd.concat(df_list, ignore_index=True)


# print(sum(df['amount_fiat']))
# df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
# print(df.head())

# summarize the data
# print(df.describe())

df['completed_at'] = pd.to_datetime(df['completed_at'])
df['year'] = df['completed_at'].dt.year
df['month'] = df['completed_at'].dt.month
df['day'] = df['completed_at'].dt.day

df_grouped = df.groupby(['year', 'month', 'day']).agg({
                         'amount_fiat': 'mean'}).reset_index()

# Create a new column 'date_numeric' for linear regression
# Convert year, month, day into a numerical format (e.g., days since the first day)
df_grouped['date_numeric'] = pd.to_datetime(df_grouped[['year', 'month', 'day']])
df_grouped['date_numeric'] = (df_grouped['date_numeric'] -
                              df_grouped['date_numeric'].min()).dt.days

# Print the updated dataframe
print(df_grouped.head(len(df_grouped)))


# CREATE AND TRAIN DATASET

msk = np.random.rand(len(df_grouped)) < 0.8
train = df_grouped[msk]
test = df_grouped[~msk]

# Simple Regression Model

# Training
plt.scatter(train['date_numeric'], train['amount_fiat'], color='blue')
plt.xlabel("Days since first trade")
plt.ylabel("Average Fiat Amount")
plt.show()

# Modeling with linear regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['date_numeric']])
train_y = np.asanyarray(train[['amount_fiat']])
regr.fit(train_x, train_y)

# Print coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

average_daily_amount = df_grouped['amount_fiat'].mean()
print("Average fiat amount made daily: %.2f" % average_daily_amount)


# EVALUATE MEAN ABSOLUTE ERROR, MEAN SQUARED ERROR, R2-SCORE

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['date_numeric']])
test_y = np.asanyarray(test[['amount_fiat']])
test_y_ = regr.predict(test_x)
print("Range of Y prediction: {}".format(test_y_))

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
