# Sparsify Neurons

This project aims to reduce the numberof neurons in various neural networks using variational dropout technique discussed in [this paper](https://arxiv.org/abs/1701.05369).

The method in the paper sparsifies the weights of neural networks very well, which leads to a good compression. However, they do not achieve reducing the number of neurons, while such result would significantly speed up the inference stage.

## Method

I decided to use the same approach, with only difference in the prior distribution. For the fully-connected linear layers $x = Wy$, where $W$ is $m \times n$ matrix, I set the prior distribution

$$p(W|a, b) = \prod\limits_{i, j}\mathcal{N}(W_{ij}|0, a_ib_j)$$

This way, I give a scale factor for all input and output neurons, so that all weights in a row/column with low scale will tend to be nullified. On the other hand, if many rows/columns have high scale factors, the ELBO will be low. Thereore, in the optimum many rows and columns have low scale factor, making the corresponding neurons irrelevant for the predictions.

During optimization, I use the same posterior distribution estimation, as in the paper: $q(W) = \prod\limits_{i, j} \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$. Then

$$KL(q||p) = \frac{1}{2}\sum\limits_{ij} \left(\log\frac{a_ib_j}{\sigma_{ij}^2} + \frac{\mu_{ij}^2 + \sigma_{ij}^2}{a_ib_j} - 1\right)$$

Using derivatives, we can see that the minimum wrt $a, b$ is achieved when

$$a_i = \frac{1}{n} \sum\limits_{j} \frac{\mu_{ij}^2 + \sigma_{ij}^2}{b_j},\ b_j = \frac{1}{m} \sum\limits_{i} \frac{\mu_{ij}^2 + \sigma_{ij}^2}{a_i}$$

During the ELBO computation, I compute the scale factors $a, b$ iteratively, starting with initial values of all ones, and then applying the above formulas. I checked that 3 iterations is enough to converge to stable solution. After finding the optimal $a, b$, I substitute them in the KL to get

$$KL(q||p) = \frac{1}{2}\sum\limits_{ij}\log\frac{a_ib_j}{\sigma_{ij}^2}$$

The computations for the convolutional layers are very similar, with scale factors corresponding to every input and output channels. I believe, such computations can be generalized for any weight-based linear transform, e.g. multi-head attention. I've created a single class BayesianLinear(nn.Module) (./utils/bayesian_nn.py), where created a framework to generalize such bayesian layer quite easily for any linear transform.


