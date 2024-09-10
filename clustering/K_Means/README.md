## K-Means Clustering

`` Despite its simplicity, the K-means is vastly used for clustering in many data science applications, it is especially useful if you need to quickly discover insights from unlabeled data.``


Some real-world applications of k-means:

- [x] Customer segmentation
- [x] Understand what the visitors of a website are trying to accomplish
- [x] Pattern recognition
- [x] Machine learning
- [x] Data compression


``The make_blobs class can take in many inputs, but we will be using these specific ones.``

#### Input

- [x] n_samples: The total number of points equally divided among clusters.
	- [x] Value will be: 5000
- [x] centers: The number of centers to generate, or the fixed center locations.
	- [x] Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
- [x] cluster_std: The standard deviation of the clusters.
	- [x] Value will be: 0.9

#### Output

- [x] X: Array of shape [n_samples, n_features]. (Feature Matrix)
	- [x] The generated samples.
- [x] y: Array of shape [n_samples]. (Response Vector)
	- [x] The integer labels for cluster membership of each sample.


### Setting up K-Means

``The KMeans class has many parameters that can be used, but we will be using these three:``

- [x] init: Initialization method of the centroids.
	- [x] Value will be: "k-means++"
	- [x] k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.

- [x] n_clusters: The number of clusters to form as well as the number of centroids to generate.
	- [x] Value will be: 4 (since we have 4 centers)

- [x] n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
	- [x] Value will be: 12


## Customer Segmentation with K-Means

``Imagine that you have a customer dataset, and you need to apply customer segmentation on this historical data. Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. For example, one group might contain customers who are high-profit and low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retain those customers. Another group might include customers from non-profit organizations and so on.``


