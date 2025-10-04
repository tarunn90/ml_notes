
# Related Links
- [CS 231N course notes](https://cs231n.github.io/optimization-1/) 


# Optimization
The **numerical gradient** is simply approximating the derivative by taking a very small step in the direction of each dimension and calculating the slope of the change:

$$
\frac{df(x)}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}
$$

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

# Activation Functions

## Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Squashes real numbers into the range $[0,1]$ 
- Rarely ever used in practice for 2 reasons:
	- Sigmoids *saturate and kill gradients*, restricting the flow of signal further up the network
	- Sigmoid outputs are not zero-centered

## Tanh

$$\text{tanh}(x) = 2\sigma(2x) - 1$$

- Squashes real numbers into the range $[-1,1]$ 
- Simply a **zero-centered** and scaled sigmoid function

# ReLU

$$\text{ReLU}(x) = \max(0,x)$$

- Pros
	- Speeds up convergence during gradient descent because it doesn't saturate the gradient unlike sigmoid/tanh
	- Simpler to calculate forward-pass compared to sigmoid/tanh
- Con
	- Can "die" during training if the gradient gets stuck at zero

## Leaky ReLU

$$\text{Leaky ReLU}(x) = 1(x < 0)(\alpha x) + 1(x \geq 0)(x)$$

- Attempt to fix the "dying ReLU" problem
- Has small positive slope when $x < 0$ 

## Maxout

$$\text{Maxout}(x) = \max(w_1^Tx+b_1, w_2^Tx + b_2)$$
- Has benefits of ReLU but without "dying" problem
- Doubles the number of parameters

## Summary

“_What neuron type should I use?_” Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of “dead” units in a network. If this concerns you, give Leaky ReLU or Maxout a try. Never use sigmoid. Try tanh, but expect it to work worse than ReLU/Maxout.


# Architecture

- An $N$ layer neural network includes the output but *not* the input layer, so a 1-layer neural network maps the input to the output with no hidden layer and a 2-layer neural network has one hidden layer
- The output layer does not usually have an activation function since this usually maps to the class scores or regression output


## Universal Approximation

Neural networks with at least one hidden layer are *universal approximators*: 

Given any continuous function $f(x)$ and $\epsilon > 0$ there exists a neural network $g(x)$ with one hidden layer and some non-linearity such that $\forall x, |f(x) - g(x)| < \epsilon$ 

In practice, a two-layer NN is pretty weak and useless. In practice, more layers gives equal representational power for fewer parameters. 

## Overfitting

The more neurons and the more hidden layers we have, the more likely we are to overfit. However, it is better to keep the model complex but use **regularization** to prevent overfitting. 

> The subtle reason behind this is that smaller networks are harder to train with local methods such as Gradient Descent: It’s clear that their loss functions have relatively few local minima, but it turns out that many of these minima are easier to converge to, and that they are bad (i.e. with high loss). Conversely, bigger neural networks contain significantly more local minima, but these minima turn out to be much better in terms of their actual loss. Since Neural Networks are non-convex, it is hard to study these properties mathematically, but some attempts to understand these objective functions have been made, e.g. in a recent paper [The Loss Surfaces of Multilayer Networks](http://arxiv.org/abs/1412.0233). In practice, what you find is that if you train a small network the final loss can display a good amount of variance - in some cases you get lucky and converge to a good place but in some cases you get trapped in one of the bad minima. On the other hand, if you train a large network you’ll start to find many different solutions, but the variance in the final achieved loss will be much smaller. In other words, all solutions are about equally as good, and rely less on the luck of random initialization.


# Data Preprocessing

- **Mean subraction:** subtract the mean from every feature in the data. 
	- Every feature looks like a cloud around the origin
- **Normalization:** two methods
	- Zero-center and divide by standard deviation: `X /= np.std(X, axis=0)` 
	- OR min-max scale each feature to lie in $[-1,1]$ 

These scaling methods only make sense if each feature has different scale but should be of equal relative importance during gradient descent. For images, we don't need to do any scaling since every pixel value already lies in $[0, 255]$ 

- **PCA dimensionality reduction:** take the first K components from PCA after zero-centering the data. 

# Weight Initialization

- DON'T: initialize all weights to 0. If every neuron computes the same output, then they will all get the same gradients during backprop and get the same updates. 
- **Small random numbers:** with proper normalization, we can guess that the weights will be evenly distributed around the origin. 
	- `W = 0.01 * np.random.randn(D, H)` where `D` = number of neurons in current layer and `H` = number of neurons in previous layer 
- **Scale by sqrt(n)**: If we follow the above approach, the variance of the outputs scales with the number of inputs. Therefore, we can inverse-weight by the number of inputs: `w = np.random.randn(n)/sqrt(n)`. This ensures that neurons with more inputs start off an an even footing with other neurons. 
- **Zero bias** initialization is common for biases
- **Batch normalization** actually takes care of a lot of the initialization headaches for us! 
	- Batch normalization is differentiable and forces weights to take on a unit gaussian distribution from the start of training

Let $m$ be batch size:
$$
\displaylines{
\mu = \frac{1}{m} \sum_i x_i \\
\sigma^2 = \frac{1}{m} \sum_i (x_i - \mu) \\
\hat{x} = (x_i - \mu)/(\sqrt{\sigma^2} + \epsilon) \\
\text{BatchNorm}(x) = \gamma \hat{x} + \beta
}
$$

where $\gamma, \beta$ are learnable parameters (scale and shift respectively) 

# Regularization

## L2 Regularization

We add the term $\frac{1}{2} \lambda \sum_j w_j^2$ to the loss function. This encourages weights to be more diffuse across neurons rather than "peaky". 

Works better than L1 if you don't want feature selection. 

## L1 Regularization

We add the term $\lambda \sum_j |w_j|$ to the loss function

## ElasticNet

Combine L1 and L2 with a tunable parameter $\alpha$: 

$\alpha |w| + (1 - \alpha) w^2$  

## Dropout 

With probability $p$ each neuron is set to zero. 

The vanilla implementation sets the weights to zero with probability $p$ at train time but then needs to scale them down by $p$ at predict time: 

```
""" Vanilla Dropout: Not recommended implementation """

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  """ X contains the data """
  
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3
```

Instead, we use **inverted dropout**, where we drop *and* scale at train time: 

```
""" 
Inverted Dropout: Recommended implementation example.
We drop and scale at train time and don't do anything at test time.
"""

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```


# Loss Functions

## Classification

* Cross-entropy loss

## Regression

* L2 norm loss: $||f - y_i||^2$
	* equivalent to MSE
* L1 norm loss: $|f - y_i|$ 
	* Note that the gradient for absolute value is $\text{sign}$ 

The L2 norm loss is not robust to outliers: these can introduce huge gradients. Therefore, it can be worth transforming the problem from regression into classification, e.g., binning the outcomes. 


# Training

## Gradient Checks
Use the centered numerical gradient:
$$\frac{df(x)}{dx} = \frac{f(x+h) - f(x-h)}{2h}$$
Check the relative error between the analytic and numerical gradients:
$$\frac{|f'_a - f'_n|}{\max(|f'_a|, |f'_n|)}$$
Note that precision really matters here: use double not single floating point precision.

## Metrics to Track

**Loss function**
<img src="imgs/Pasted image 20251003123724.png"> 

**Train/Validation accuracy**
<img src="imgs/Pasted image 20251003123901.png"> 


# Parameter Updates

Let $w$ be the weight and $dw$ be the gradient after backpropagation.

## **Vanilla updates:** 
$w = w - \alpha (dw)$ where $\alpha$ is the learning rate
```python 
w -= alpha * dw
```

## **Momentum update**

For momentum, we introduce another variable: $v$, the velocity. Here, $v$ is a stateful variable (initialized at 0) that changes in proportion to the gradient. In turn, the velocity informs the position of the weights. 

$$ 
\displaylines{
v = \mu v - \alpha (dw) \\
w = w + v
}
$$
$\mu$ here is similar to the coefficient of friction: it tell us how much to dampen the velocity. Typically we anneal it over time, starting at 0.5 and ending at 0.99. Confusingly, $\mu$ is called "momentum". 

```python 
v = mu * v - alpha * dw
w += v
```

## Annealing the Learning Rate

- **Step decay:** reduce hte learning rate by some factor every few epochs, e.g., by 0.1 every 20 epochs. 
- **Exponential decay:** $\alpha = \alpha_0 e^{-kt}$ where $t$ = number of epochs. 
- **1/t decay**: $\alpha = \alpha_0 / (1 + kt)$ 

## Second Order Methods

With second order methods (based on Newton's method) we can perform more aggressive parameter updates by considering the curvature of the loss function. A popular method here is **L-BFGS** but it is quite computationally expensive because it doesn't work well on mini-batches and so it isn't common to use. 

## Adagrad

Adagrad is an adaptive learning rate method which is run **per-parameter**. 

```python
cache += dw**2 
w += -alpha * dw / (np.sqrt(cache) + eps)
```

`cache` has size equal to the size of the gradient, and has the per-parameter sum of squared gradients. This way, parameters with large gradients will have smaller learning rates while parameters with small updates will have higher learning rates. 

## RMSProp

RMSProp uses a moving average of squared gradients:
```python 
cache += decay_rate * cahce + (1 - decay_rate * dw **2)
w -= alpha * dx / (np.sqrt(cache) + eps)
```

The cache variable here is "leaky": the updates do not get monotonically smaller. 

## Adam
Adam is similar to RMSProp but with momentum 

```python 
m = beta1*m + (1-beta1)*dw
v = beta2*v + (1-beta2)*(dw**2)
w -= alpha * m / (np.sqrt(v) + eps)
```

It has a "smooth"version of the gradient from RMSProp. 
# Hyperparameter Optimization

- Random search > Grid Search
- Stage the search from corase to fine
- Bayesian HPO: [Hyperopt](https://jaberg.github.io/hyperopt/) 