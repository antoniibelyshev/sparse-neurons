# Abstract

In this project, I aim to sparsify the neurons of neural networks by extending the variational dropout technique introduced in the [Variational Dropout paper](https://arxiv.org/abs/1701.05369). While the original approach successfully sparsifies weights, it does not reduce the number of neurons. By proposing a modified prior distribution that encourages neuron sparsification, I achieve significant neuron reduction in fully connected and convolutional layers, leading to more efficient networks.

Key findings include:
- **LeNet-300-100 on MNIST**: Neuron count reduced by **72.25%**, with a minor drop in accuracy from **98.58% to 98.08%**.
- **VGG-16 on CIFAR-10**: Neuron count reduced by **80.89%**, with accuracy decreasing slightly from **88.89% to 87.40%**.
  
This approach holds potential for improving inference efficiency in deep neural networks by reducing neuron redundancy while maintaining competitive accuracy.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Method](#method)
3. [Optimization](#optimization)
4. [Experiments](#experiments)
5. [Results and Conclusion](#results-and-conclusion)

---

## Introduction

### Project Context
Neural networks have grown increasingly large, which boosts their capacity but also increases computational costs, particularly during the inference stage. This is an issue for real-time applications or devices with limited hardware resources, such as mobile phones or embedded systems. Traditional weight sparsification techniques, such as variational dropout, address this by eliminating unnecessary weights, but they do not remove redundant neurons, which is essential for maximizing inference efficiency.

### Goal of the Project
The aim of this project is to extend the variational dropout technique to not only sparsify weights but also eliminate redundant neurons. By introducing a novel prior distribution, we can significantly reduce the number of neurons in the network, thereby improving computational efficiency while keeping model accuracy high. The focus is on sparsifying both fully connected layers and convolutional layers.

---

## Method

### Prior Work: Variational Dropout
The method for sparsifying weights in neural networks is based on the Variational Dropout technique. In this approach, the prior distribution of weights $p(W)$ is set to be a centered Gaussian distribution. The goal is to optimize approximation of posterior distribution to match the true posterior by maximizing the Evidence Lower Bound (ELBO):

$$\mathcal{L}(q, \theta) = \mathbb{E}_q \log (Y|W, X) - KL(q||p)$$

This technique encourages many weights to become small and eventually zero, allowing for efficient weight sparsification.

### Approach

#### Posterior Distribution
I follow the standard approach of approximating the posterior distribution of weights $q(W)$ with Gaussian distributions. The mean $\mu_{ij}$ and variance $s_{ij}^2$ of the weights are learned during training:

$$q(W) = \prod\limits_{ij}\mathcal{N}(W_{ij}|\mu_{ij}, s_{ij}^2)$$

#### Prior Distribution
While the traditional variational dropout uses a prior distribution that focuses on sparsifying individual weights, my goal is to extend this to neuron sparsification. To do this, I modify the prior distribution. Instead of assigning a different scale for each weight, I use the same scale for all weights associated with a given output neuron:

$$p(W|\sigma) = \prod_{ij}\mathcal{N}(W_{ij}|0, \sigma_i^2)$$

This forces all weights connected to a neuron to either remain relevant (non-zero) or become irrelevant (close to zero), thus leading to neuron sparsification.

#### Mixture of Gaussians
Since this approach can be overly restrictive, I extend the prior distribution to a mixture of two Gaussians:

$$p(W|\theta) = \prod_{ij}\left( p_i\mathcal{N}(W_{ij}|0, {\sigma_1}_i^2) + (1 - p_i)\mathcal{N}(w_{ij}|0, {\sigma_2}_i^2) \right)$$

This allows neurons to have a mixture of large and small weights, better capturing the behavior of different layers and improving model accuracy while encouraging neuron sparsification.

---

## Optimization

I focus on maximizing the ELBO w.r.t. $q$ and $\theta$. I start with maximizing it w.r.t. $\theta$. In the ELBO, only the $KL$ term depends on $\theta$:

$$-KL(q||p) = -\mathbb{E}_q\log\frac{q(W)}{p(W|\theta)} = \mathbb{E}_q \log p(W|\theta) + const$$

Therefore, I need to maximize $\mathbb{E}_q\log p(W|\theta)$ w.r.t. $\theta$. I show in the appendix (./Appendix/gaussian_mixture.pdf) that $\sigma_1, \sigma_2, p$ can be found by running an EM algorithm for a Gaussian mixture model with two centered Gaussian, assuming the observed samples are $\tilde W_{ij} = \sqrt{\mu_{ij}^2 + s_{ij}^2}$. For convergence, one EM step per training step is sufficient. I also demonstrate that after optimizing $\theta$,

$$\mathbb{E}_q \log p(W|\theta) \ge \log p(\tilde W|\theta)$$

where $\tilde W_{ij} = \sqrt{\mu_{ij}^2 + s_{ij}^2}$. Combining all parts of the $KL$ term, we get:

$$KL(q||p) = \mathbb{E}_q \log q - \mathbb{E}_q \log p \le -\frac{1}{2}\sum_{ij} \log(2\pi es_{ij}^2) - \log p(\tilde W|\theta)$$

While this expression is an approximation, it is easy to compute and minimizing it will also minimize $KL$.

### Optimization Step
Each optimization step consists of the following parts:

1. Perform a forward pass through the network using the reparameterization trick in each linear layer (this forward pass is explained in the article I follow).
2. Compute the cross-entropy $\log p(Y|W, \theta)$.
3. Update $\theta$ using one EM step.
4. Compute an upper bound for $KL(q||p)$.
5. Compute the ELBO, calculate its gradient, and update the weights.

### Neuron Sparsification
As explained in the Variational Dropout article, many weights will have a high dropout rate after training. These weights will be almost certainly zero, and therefore, such weights are irrelevant. I consider a weight irrelevant if its dropout rate is higher than 0.99. In terms of neurons, they are irrelevant if all corresponding weights are irrelevant, because then the neuron activation will most probably be zero.

### Generalization for Different Types of Layers
I explained the approach for a fully connected linear layer. However, it can be easily generalized for other types of linear transforms. Such generalization for convolutional layers is already discussed in the article I followed. Moreover, I believe that such ideas can be applied to other types of linear transforms, e.g., Multi-Head Attention.

---

## Experiments

### Setup
I conducted experiments with LeNet-300-100 applied to MNIST and VGG-16 applied to CIFAR-10. All experiments can be reproduced using `train.py` from the terminal:

- **LeNet**:
  ```bash
  python3 train.py meta.model=LeNet meta.dataset=MNIST meta.prefix=""
  ```

- **LeNet with Bayesian fully-connected layers**:
  ```bash
  python3 train.py meta.model=LeNet meta.dataset=MNIST meta.prefix=Bayesian
  ```

- **VGG**:
  ```bash
  python3 train.py meta.model=VGG meta.dataset=CIFAR meta.prefix=""
  ```

- **VGG with Bayesian convolutional layers**:
  ```bash
  python3 train.py meta.model=VGG meta.dataset=CIFAR meta.prefix=Bayesian
  ```

All hyperparameters are set in the config files located in the `./config/` directory. In order to run the code, you need python==3.10 and packages specified in the requirements.txt.

### Results

#### LeNet-300-100 on MNIST
The results of applying neuron sparsification to LeNet-300-100 are as follows:

- **Default model accuracy**: 98.58%
- **Bayesian model accuracy**: 98.08%

**Neuron sparsification results**:

| Layer number | Default LeNet config | Squeezed LeNet config |
|--------------|----------------------|-----------------------|
| 1            | 300                  | 56                    |
| 2            | 100                  | 55                    |

**Total neuron sparsification**: **72.25%** (111/400)

#### VGG-16 on CIFAR-10
The results of applying neuron sparsification to VGG-16 are as follows:

- **Default model accuracy**: 88.89%
- **Bayesian model accuracy**: 87.40%

**Neuron sparsification results**:

| Layer number | Default VGG config | Squeezed VGG config |
|--------------|--------------------|---------------------|
| 1            | 64                 | 56                  |
| 2            | 64                 | 60                  |
| 3            | M                  | M                   |
| 4            | 128                | 120                 |
| 5            | 128                | 122                 |
| 6            | M                  | M                   |
| 7            | 256                | 180                 |
| 8            | 256                | 66                  |
| 9            | 256                | 45                  |
| 10           | M                  | M                   |
| 11           | 512                | 27                  |
| 12           | 512                | 19                  |
| 13           | 512                | 29                  |
| 14           | M                  | M                   |
| 15           | 512                | 21                  |
| 16           | 512                | 27                  |
| 17           | 512                | 35                  |
| 18           | M                  | M                   |

**Total neuron sparsification**: **80.89%** (3417/4224)

---

## Results and Conclusion

### Key Findings
- Using the proposed variational dropout method with a mixture of Gaussian priors, I achieved significant neuron sparsification in both LeNet-300-100 and VGG-16.
- **LeNet** saw a **72.25%** reduction in neurons, while **VGG-16** achieved a **80.89%** reduction.
- The accuracy drop was minimal, with only a **0.5%** reduction in LeNet and **1.49%** in VGG-16, showing that sparsification did not severely affect the model's performance.

### Future Work
In the future, I aim to extend this approach to additional architectures and more complex datasets. Furthermore, incorporating neuron sparsification into attention mechanisms and recurrent layers could offer new directions for optimizing deep learning models in various domains.

Another direction ow work would be to try other prior distributions which could possibly restrict the weights less. A specific distribution which I want to try is t-distribution. Intuitively, it should restrict large values of weights less than Gausian or Gaussian mixture. That could result in better accuracy. Additionally, the computation of KL divergence could be computed in similar way, by considering EM algorithm for fitting t-distribution.

---
