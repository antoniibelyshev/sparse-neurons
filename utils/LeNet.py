import torch
from torch import nn, Tensor

from .bayesian_nn import BayesianLinear


class LeNet(nn.Module):
    """
    Standard LeNet model for fully connected layers.

    Args:
        in_features (int): Input feature size. Default is 784.
        num_classes (int): Number of output classes. Default is 10.
        cfg (list[int]): List of hidden layer sizes. Default is [500, 300].
        bias (bool): Whether to use bias in layers. Default is True.
    """
    def __init__(self, in_features: int = 784, num_classes: int = 10, cfg: list[int] = [300, 100], bias: bool = True):
        super(LeNet, self).__init__()  # type: ignore

        self.layers = nn.ModuleList()

        for out_features in cfg:
            self.layers.append(nn.Linear(in_features, out_features, bias))
            self.layers.append(nn.ReLU())
            in_features = out_features

        self.layers.append(nn.Linear(in_features, num_classes, bias))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class BayesianLeNet(nn.Module):
    """
    Bayesian version of LeNet with uncertainty modeling.

    Args:
        in_features (int): Input feature size. Default is 784.
        num_classes (int): Number of output classes. Default is 10.
        cfg (list[int]): List of hidden layer sizes. Default is [500, 300].
        bias (bool): Whether to use bias in layers. Default is True.
        threshold (float): Dropout threshold for Bayesian layers. Default is 0.99.
    """
    def __init__(self, in_features: int = 784, num_classes: int = 10, cfg: list[int] = [500, 300], bias: bool = True, threshold: float = 0.99):
        super(BayesianLeNet, self).__init__()  # type: ignore

        self.in_features = in_features
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        for out_features in cfg:
            self.layers.append(BayesianLinear(in_features, out_features, bias=bias, threshold=threshold, linear_transform_type='linear'))
            self.layers.append(nn.ReLU())
            in_features = out_features

        self.layers.append(BayesianLinear(in_features, num_classes, bias=bias, threshold=threshold, linear_transform_type='linear'))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Bayesian network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def kl(self) -> Tensor:
        """
        Computes the KL divergence for all Bayesian layers.

        Returns:
            Tensor: Total KL divergence.
        """
        kl = torch.zeros(1, device=self.device)
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl()
        return kl
    
    @property
    def device(self) -> torch.device:
        """
        Returns the device the model is on.
        """
        return next(self.parameters()).device
    
    def get_bayesian_layers(self) -> list[BayesianLinear]:
        """
        Returns a list of all Bayesian layers.

        Returns:
            list[BayesianLinear]: List of BayesianLinear layers.
        """
        return [layer for layer in self.layers if isinstance(layer, BayesianLinear)]
    
    def squeezed_cfg(self) -> list[int]:
        """
        Returns the configuration of the squeezed network based on dropout.

        Returns:
            list[int]: List of output sizes for each layer.
        """
        return [int(layer.get_out_mask().sum()) for layer in self.get_bayesian_layers()]
    
    def squeeze(self) -> LeNet:
        """
        Converts the Bayesian network into a standard LeNet with the squeezed configuration.

        Returns:
            LeNet: The squeezed LeNet model.
        """
        cfg = self.squeezed_cfg()
        lenet = LeNet(self.in_features, self.num_classes, cfg)

        for layer, layer_ in zip(lenet.layers, self.layers):
            if isinstance(layer, nn.Linear) and isinstance(layer_, BayesianLinear):
                layer.weight.data = layer_.weight.data
                if layer_.bias is not None:
                    layer.bias.data = layer_.bias.data

        return lenet
