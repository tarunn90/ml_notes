# Model Structure

Define $K$ centroids (points); each represents a distinct cluster. Assign all points to the nearest centroid, as measured by given distance metric.

# Objective function

Objective function for one cluster: **minimize the** **_within-cluster variation_** (given by sum of pairwise squared Euclidean distance) between all the points in that cluster:

  ![[WIG)- CA 2 Eli, - cA)2,.png]]


Objective function across clusters: minimize the within-cluster variation across all clusters

![[minimize 1.png]]

Or:

![[minimize 2.png]]
  

The within-cluster variation is the same as the Euclidean distance of all the points to each point’s centroid:

![[TON E Ele, w73=2 ELoy.png]]

# Algorithm

**Note that we are *not guaranteed* to reach the global optimum**

1. Initialize centroids $1,...,K$ from the dataset at random
2. Until the cluster assignments stop changing:
	- Assign each point to its nearest centroid: these are the new cluster assignments
	- Re-compute the centroids by calculating the centroid of each new cluster

# Assumptions and Weaknesses
- Assumes spherical clustering of data
- Euclidean distance might be meaningless in high dimensions: curse of dimensionality
- Might not converge to global optimum
- Sensitive to noise from outliers
## Initialization
- Results are heavily dependent on the value of $K$ 
- K-means++ algorithm improves by choosing initial centroids with diversity

## Tuning K 
Best practice is to look use the **elbow method**: plot K vs WCSS (within cluster sum of squares) and find where the WCSS improvement "slows" significantly as K increases 
	- Instead of WCSS, can look at **silhouette score**: measures how far every point is from its centroid versus other centroids. Higher is better. Plot K vs silhouette score. 