#!/usr/bin/env python3
"""A credit card fraud detection script using Scikit-Learn and Snap ML
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
from snapml import SupportVectorMachine
from snapml import DecisionTreeClassifier as SMLDTC
import time
import warnings
warnings.filterwarnings('ignore')


credit_card_file = "creditcard.csv"
raw_data = pd.read_csv(credit_card_file)

print(raw_data.head())

# ~~~~ inflate the original dataset cuz it's advisable to inflate big data
# ~~~~ to stimulate. Inflate 10 times

inflate_replicas = 10
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, inflate_replicas,
                            axis=0), columns=raw_data.columns)

print(big_raw_data.head())

# range of the amounts (min/max) in the transactions
x = big_raw_data.Amount.values
print("Min amount: ", np.min(x))
print("Max amount: ", np.max(x))

# get 90th percentile of the amount values
print("90% of the transactions have an amount less or qual than: ",
      np.percentile(x, 90))

# data preprocessing
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(
                             big_raw_data.iloc[:, 1:30])

data_matrix = big_raw_data.values

X = data_matrix[:, 1:30]
y = data_matrix[:, 30]

X = normalize(X, norm="l1")

# TRAIN/TEST Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                   random_state=42, stratify=y)


# Building Decision Tree Classifier Model with Scikit-Learn
weight_train = compute_sample_weight('balanced', y_train)

sklearn_dt = DTC(max_depth=4, random_state=35)
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=weight_train)
sklearn_time = time.time()-t0
print("SKlearn training time (s): {0:.5f}".format(sklearn_time))


# Building Decision Tree Classifier Model using Snap ML
snapml_dt = SMLDTC(max_depth=4, random_state=45, n_jobs=4)
t0 = time.time()
snapml_dt.fit(X_train, y_train, sample_weight=weight_train)
snapml_time = time.time()-t0
print("SnapML training time (s): {0:.5f}".format(snapml_time))


# Build a support vector Machine Model with Scikit-Learn
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31,
                        loss="hinge", fit_intercept=False)
sklearn_svm.fit(X_train, y_train)

# Build a support vector Machine Model with Snap ML
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25,
             n_jobs=4, fit_intercept=False)
snapml_svm.fit(X_train, y_train)

# Evaluate the quality of the SVM models trained above using the hinge loss
# metric, Print the hinge lossess of Scikit-Learn and Snap ML
sklearn_pred_decision = sklearn_svm.decision_function(X_test)

print("Sklearn Hinge loss: ", hinge_loss(y_test, sklearn_pred_decision))


snapml_pred_decision = snapml_svm.decision_function(X_test)

print("SnapML Hinge Loss: ", hinge_loss(y_test, snapml_pred_decision))
