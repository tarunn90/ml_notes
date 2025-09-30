
# Related Links
- [Course notes](https://cs231n.github.io/optimization-1/) 


# Optimization
The **numerical gradient** is simply approximating the derivative by taking a very small step in the direction of each dimension and calculating the slope of the change:

<img src="Pasted image 20250929214031.png">

This is simple but *very* slow. 

The **analytic gradient** requires calculus and is more error-prone but is much faster since you're doing exact calculations over the gradient. You just calculate the partial derivative of the loss function with respect to each of the parameters in the weights vector: $$\nabla L(w) = [\frac{\delta L}{w_1},\frac{\delta L}{w_2},...]$$


Comparing the numerical gradient against the analytic gradient is a useful check in practice. 

## Gradient Descent
The following basic core loop is called **gradient descent:**
1. Evaluate the gradient with your current parameters: if $w$ is a weights vector then: $$\nabla L(w) = [\frac{\delta L}{w_1},\frac{\delta L}{w_2},...]$$
2. Update your parameters $w$ in the negative of the direction of the gradient: $$ w := w - \alpha \nabla L(w)$$

- Rather than updating one example at a time (**stochastic gradient descent**), we can do **minibatch gradient descent**. Due to parallelization and vectorization, it is faster to evaluate the gradient for 100 examples than to find the gradient for one example 100 times. 
	- Note that the batch size is often in powers of 2 because many vectorized operations work better when the input is a power of 2. 

# Backpropagation
**Standardizing** the input data during preprocessing is very important for gradient descent because otherwise the gradient descent will focus on one feature and ignore the others. 

