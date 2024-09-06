Imagine you are a data scientist working for a real estate company that is planning to invest in B real estate. You have collected information about various areas of B and are tasked with created a model that can predict the median price of houses for that area so it can be used to make offers.

The dataset had information on areas/towns not individual houses, the features are

``CRIM:`` Crime per capita

``ZN:`` Proportion of residential land zoned for lots over 25,000 sq.ft.

``INDUS:`` Proportion of non-retail business acres per town

``CHAS:`` Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

``NOX:`` Nitric oxides concentration (parts per 10 million)

``RM:`` Average number of rooms per dwelling

``AGE:`` Proportion of owner-occupied units built prior to 1940

``DIS:`` Weighted distances to ﬁve Boston employment centers

``RAD:`` Index of accessibility to radial highways

``TAX:`` Full-value property-tax rate per $10,000

``PTRAIO:`` Pupil-teacher ratio by town

``LSTAT:`` Percent lower status of the population

``MEDV:`` Median value of owner-occupied homes in $1000s


## CREATING REGRESSION TREES

Regression Trees are implemented using:

``DecisionTreeRegressor from sklearn.tree``

The important parameters of ``DecisionTreeRegressor`` are:

``criterion:`` {"mse", "friedman_mse", "mae", "poisson"} - The function used to measure error

``max_depth`` - The max depth the tree can be

``min_samples_split`` - The minimum number of samples required to split a node

``min_samples_leaf`` - The minimum number of samples that a leaf can contain

``max_features:`` {"auto", "sqrt", "log2"} - The number of feature we examine looking for the best one, used to speed up training
