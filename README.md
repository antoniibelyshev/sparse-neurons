# Neural Network Neuron Sparsification via Variational Dropout

## Overview

In this project I extend the variational dropout method to sparsify not only weights but also neurons in neural networks, leading to a reduction in computational costs without sacrificing much accuracy. Neural networks often have redundancy in their architecture, which increases the computational resources, particularly during inference. The standard variational dropout technique focuses on reducing unnecessary weights but does not explicitly target redundant neurons. Here, I propose a modified prior distribution to achieve significant neuron reduction.

### Key Results

1. **LeNet-300-100 on MNIST**: 
   - **Neuron reduction**: 72.25%
   - **Accuracy**: Minor drop from 98.58% to 98.08%
   
2. **VGG-16 on CIFAR-10**: 
   - **Neuron reduction**: 80.89%
   - **Accuracy**: Slight decrease from 88.89% to 87.40%

### Method Overview

The approach builds on the Variational Dropout technique, which aims to sparsify weights using a Gaussian prior distribution. The primary extension in this project modifies the prior distribution to focus on neuron sparsification, ensuring that all weights connected to an irrelevant neuron are encouraged to be zero or close to zero. This modification helps identify and remove redundant neurons, thereby reducing the network size.

Additionally, a mixture of two Gaussian distributions is used as a prior, allowing neurons to have either small or large weights, providing flexibility while still promoting neuron sparsification.

### How to Run the Code

You can reproduce the results using the following commands:

- **Raw LeNet-300-100**:
  ```bash
  python3 train.py meta.model=LeNet meta.dataset=MNIST meta.prefix=""
  ```
- **Bayesian LeNet-300-100**:
  ```bash
  python3 train.py meta.model=LeNet meta.dataset=MNIST meta.prefix=Bayesian
  ```
- **Raw VGG-16**:
  ```bash
  python3 train.py meta.model=VGG meta.dataset=CIFAR meta.prefix=""
  ```
- **Bayesian VGG-16**:
  ```bash
  python3 train.py meta.model=VGG meta.dataset=CIFAR meta.prefix=Bayesian
  ```
All specifics about hyperparameters please refer to the configuration files in the ```.config/``` directory.

To run the code, you need ```python3.10``` and packages specified in ```requirements.txt```.

### Conclusion

This project successfully demonstrates that neuron sparsification is feasible with minimal impact on model accuracy, achieving reductions of over 70% in both LeNet and VGG-16 architectures. This opens up possibilities for more efficient networks, particularly useful for real-time applications or devices with limited resources.

For further information, including theoretical derivations, please refer to the additional file with detailed theory.