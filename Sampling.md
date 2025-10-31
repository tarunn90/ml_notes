# Links
- https://www.cloudresearch.com/resources/guides/sampling/pros-cons-of-different-sampling-methods/




# Sampling High-Dimensional Data
- **Stratified Sampling:** Effective for categorical data, but fails when many categories
- **Cluster-Based Sampling:** Run clustering algorithm (K-Means, DBSCAN), then sample within each cluster
- **Density-Based Sampling:** Sample more heavily from dense areas, less heavily from sparse areas. Estimate local density using KNN or Kernel Density Estimation. 
- **Core-Set Sampling:** Select points that approximate the entire dataset for a given task, e.g., clustering, regression, classification. The idea is that if you used *only* these points, they would give roughly the same result. 
	- We choose $k$ points such that the **largest distance** from any data point to its nearest sample point is minimized, aka "**k-center problem**"
	- Formally, choose subset $S$ such that this objective is minimized: $\max\limits_{x \in X} \min\limits_{s \in S} ||x-s||$ 
- **Diversity Sampling:**, e.g., **Maxi-Min Sampling:** Iteratively select points such that each point maximizes the distance to previously selected points. This spreads points throughout space. 




# Importance Sampling
Used when you can't directly sample from a distribution $p(x)$ but:
1. You can sample from another distribution $q(x)$ 
2. You know the ratio $\frac{p(x)}{q(x)}$, i.e. the **importance weights** 

Then we sample from $q(x)$ but weight the samples by the importance weights:
$$
\frac{p(x)}{q(x)}q(x)
$$



# Imbalanced Classes
- Over/under-sampling
- Weight the classes in loss function
- SMOTE: create synthetic examples from the minority class
	- Find KNN for each minority sample
	- Interpolate between neighbors to create new example