#!/usr/bin/env python3
"""A python script that uses the K-means clustering algorithm on a randomly
generated dataset
"""


import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import warnings


def warn(*args, **kwargs):
    pass


# create a dataset for this exercise using np()
# set a random seed to create the dataset, seed is at 0
np.random.seed(0)

# next make random clusters of points by using the make_blobs class
X, y = make_blobs(n_samples=5000,
                  centers=[[4,4],
                           [-2, -1],
                           [2, -3],
                           [1, 1]],
                  cluster_std=0.9)

# display the scatter plot for the randomly generated data
plt.scatter(X[:, 0], X[:, 1], marker='.')

# set up k-means
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# fit the k-means model with the feature matrix "X"
k_means.fit(X)
print(k_means.fit(X))

# grab the labels for each point in the model using K-Means.labels_ attribute
k_means_labels = k_means.labels_
print(k_means_labels)

# get the coordinates of the cluster centers using K-Means.cluster_centers_
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)

# ~~~~ CREATING THE VISUAL PLOT ~~~~
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


"""In this section a customer dataset will be splited into different segments
using K-Means algorithm
"""

# load data from csv file
csv_data = pd.read_csv("Cust_Segmentation.csv")
print(csv_data.head())

# Address is a categorical variable and K-means algorithm isn't applicable to
# categorical variables because the Euclidean distance function isn't
# really meaningful for discrete variables, so we drop it
df = csv_data.drop('Address', axis=1)
print(df.head)

# normalize the dataset,
# Normalization is a statistical method that helps mathematical-based
# algorithms to interpret features with different magnitudes and
# distributions equally. We use StandardScaler() to normalize our dataset.
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

# Apply K-Means to the dataset
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

# We assign the labels to each row in the dataframe.
df["Clus_km"] = labels
df.head(5)

# check the centroid values by avergaing the features in each cluster
df.groupby('Clus_km').mean()

# Plot a distribution of customers based on their age and income
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

# plot 3d distrubtuion based on Education, Age and Income
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

warnings.warn = warn()
