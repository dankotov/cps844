# Imports
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt

# 1. (0 points) Load the dataset
moviesDataset = pd.read_csv("./DataLab7.csv")

# 2. (5 points) To perform a k-means analysis on the dataset, extract only the numerical attributes: remove the "user" attribute
data = moviesDataset.drop(moviesDataset.select_dtypes(exclude="number"), axis=1)

## Suppose you want to determine the number of clusters k in the initial data 'data' ##
# 3. (5 points) Create an empty list to store the SSE of each value of k (so that, eventually, we will be able to compute the optimum number of clusters k)
# decided to use a dictionary instead of a list to have the number of clusters k as a key and the respective SSE as the value
# sses = {k: sse}
sses = {}

# 4. (30 points) Apply k-means with a varying number of clusters k and compute the corresponding sum of squared errors (SSE) 
# Hint1: use a loop to try different values of k. Think about the reasonable range of values k can take (for example, 0 is probably not a good idea).
# Hint2: research about cluster.KMeans and more specifically 'inertia_'
# Hint3: If you get an AttributeError: 'NoneType' object has no attribute 'split', consider downgrading numpy to 1.21.4 this way: pip install --upgrade numpy==1.21.4

# starting range at 1 since we can not have 0 clusters
# ending range at number of user entries in the dataset since there is no point in having more clusters than there are users
# also, the python range function is exclusive of the stop number, so we would also not consider k = number of users
# this fits our purpose since it would also not make sense for each user entry to have its own cluster
for k in range(1, len(data.index)):
    kmeans = cluster.KMeans(n_clusters=k).fit(data)
    sses[k] = kmeans.inertia_
    
print(f"K-number - SSE value pairs:\n\t{sses}")

#  5. (20 points) Plot to find the SSE vs the Number of Cluster to visually find the "elbow" that estimates the number of clusters. (read online about the "elbow method" for clustering)
plt.figure()
plt.plot(list(sses.keys()), list(sses.values()))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Errors")
plt.show()

# 6. (10 points) Look at the plot and determine the number of clusters k (read online about the "elbow method" for clustering)
# for some weird reason the pyplot sets tickers of the x-axis (number of clusters values) to floats even though the actual input is only ints
# however, it is clear that the value of K = 2 is the elbow, hence its the optimal number of clusters
k = 2

# 7. (30 points) Using the optimized value for k, apply k-means on the data to partition the data, then store the labels in a variable named 'labels'
# Hint1: research about cluster.KMeans and more specifically 'labels_'
kmeans = cluster.KMeans(n_clusters=k).fit(data)
labels = kmeans.labels_

# 8. Display the assignments of each users to a cluster 
clusters = pd.DataFrame(labels, index=moviesDataset.user, columns=['Cluster ID'])
