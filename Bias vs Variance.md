The error term for any estimator can be decomposed into the bias and the variance. Let $f$ be our estimator and $y$ be the truth:
  
$$\text{Error} = \mathbb{E}[y-f] = \text{Var}(f) + \text{Bias}(f)^2$$

Here, the error denotes the expected test error we would obtain if we repeatedly learned $f$ on a large number of training sets and tested each on a given test set. 


# Variance
Variance is the amount by which $f$ would change if we estimated it using a different training data set. Ideally, $f$ would not vary too much across training sets. **However, if a given estimator has high variance then small changes in the training data can result in large changes in $f$** 

In general, more flexible methods have higher variance. For example, a regression with high-order splines will have high variance. 

# **Bias** 
Bias is the error from estimating a real-life problem with a (necessarily) simple model. 

For example, estimating the non-linear relationship between Y and X below with a linear regression will _always_ have high error, and so a linear estimator will have high bias:
<img src="Pasted Graphic.png">

If, on the other hand, the true relationship between Y and X was linear, then this estimator would have no bias, and there would be no benefit to increasing flexibility. 

High-bias models tend to make stronger assumptions about the underlying nature of the data, e.g., “X and Y have a linear relationship”.

As we increase the flexibility of our models, in general, the bias will decrease and the variance will increase. **The relative change in these quantities will determine whether the test MSE increases or decreases.** This is illustrated in the red curve of the right-hand panel below:

<img src="FIGURE 2.9. Left Data simulated from, shown in black. Three estimates on.png">

# High Variance, Low Bias

* Learns training data very well (**overfits**), doesn't generalize well
* Diagnostic: low training error, high test error
* Examples:
	* Decision trees with no pruning
	* Linear/Logistic regression with many (correlated) feautres
	* KNN with low K: low bias since it can has very complex decision boundaries, but doesn't generalize well because it is sensitive to local outliers and noise
	* Neural networks with no regularization: can learn very complex representations and can fit any function (low bias) but sensitive to initialization and training (high variance)

# High Bias, Low Variance

- Doesn't learn training data well (**underfits**), doesn't perform well on test either
- Diagnostic: performs equally poorly on train and test
- Examples:
	- Linear/Logistic regression with few features
	- KNN with large K: it has smooth decision boundaries and so is low variance, but underfits the training data so is high bias
	- Decision trees with severe pruning: simple structure => consistent across datasets