import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"E:\FSDS With GEN AI_NIT\4. April\Mall_customers.csv")
X = dataset.iloc[:,[3, 4]].values

from sklearn.cluster import KMeans

WCSS = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init="k-means++", random_state=0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1, 11), WCSS)
plt.title('The elbow method')
plt.xlabel('Number of clucsters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters= 5, init="k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

dataset['cluster'] = y_kmeans

#convert file into CSV 

dataset.to_csv('K-Means cluster.csv')

import os
os.getcwd()

