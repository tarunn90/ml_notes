# Links
- [Useful tutorial on PCA](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)

# Use Cases
- Dimensionality reduction
- Data preprocessing for linear models or neural networks
- Creating dense from sparse data
- Finding directions of most variance in the dataset

# Theory


# Algorithm


## Method 1: Use the Covariance Matrix $C$ 

**Note:** calculating $X^TX$ is very unstable in practice, so this is not usually performed. 

Let $C$ be the covariance matrix of the zero-centered matrix $\tilde{X}$:  
$$
\displaylines{
\tilde{X} = X - \mu \\
C = \frac{1}{n} \tilde{X}^T\tilde{X} \\
}
$$
Now we need to run eigendecomposition on $C$ to get its eigenvectors and eigenvalues:

$$
\displaylines{
C = V \Lambda V^T
}
$$ 
- The vectors of $V$ are the principal components
- The diagonals $\lambda_1, \lambda_2, ...$ of $\Lambda$ are the principal values 


## Method 2: Directly from $X$ 

**Note:** this method is more numerically stable and so is preferred in practice. 

We run Singular Value Decomposition on the data matrix:

$$
\displaylines{
X = U \Sigma V^T \\
}
$$

- The principal components of $X$ are the vectors of $V$ 
- The principal values are the diagonals of $\Sigma$ squared and scaled: $\lambda_i = \frac{\sigma_i^2}{n}$ 
- The projected data is given by $\hat{X} = XV$ 

# Feature Weights

The values of the principal components give the influence from each feature on that specific principal component - e.g. if there are $p$ features and $\text{PC}_1 = [\phi_1, \phi_2, ...\phi_p]$  then $\phi_j$ is the weight $\text{PC}_1$ places on the $j$-th feature. 