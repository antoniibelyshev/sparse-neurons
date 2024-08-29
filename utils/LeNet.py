import torch
from torch import nn, Tensor

from .bayesian_nn import BayesianLinear


class LeNet(nn.Module):
    def __init__(
            self,
            in_features: int = 784,
            num_classes: int = 10,
            intermediate_dims: list[int] = [500, 300],
            bias: bool = True,
    ):
        super(LeNet, self).__init__() # type: ignore

        self.layers = nn.ModuleList()

        for out_features in intermediate_dims:
            self.layers.append(nn.Linear(in_features, out_features, bias))
            self.layers.append(nn.ReLU())
            in_features = out_features

        self.layers.append(nn.Linear(in_features, num_classes, bias))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BayesianLeNet(nn.Module):
    def __init__(
            self,
            in_features: int = 784,
            num_classes: int = 10,
            intermediate_dims: list[int] = [500, 300],
            bias: bool = True,
            threshold: float = 0.99,
    ):
        super(BayesianLeNet, self).__init__() # type: ignore

        self.in_features = in_features
        self.num_classes = num_classes

        self.layers = nn.ModuleList()

        for out_features in intermediate_dims:
            self.layers.append(BayesianLinear(in_features, out_features, bias=bias, threshold=threshold, linear_transform_type='linear'))
            self.layers.append(nn.ReLU())
            in_features = out_features

        self.layers.append(BayesianLinear(in_features, num_classes, bias=bias, threshold=threshold, linear_transform_type='linear'))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def kl(self) -> Tensor:
        kl = torch.zeros(1, device=self.device)
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl()
        return kl
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_bayesian_layers(self) -> list[BayesianLinear]:
        return [layer for layer in self.layers if isinstance(layer, BayesianLinear)]
    
    def squeezed_cfg(self) -> list[int]:
        return [int(layer.get_out_mask().sum()) for layer in self.get_bayesian_layers()]
    
    def squeeze(self) -> LeNet:
        cfg = self.squeezed_cfg()

        lenet = LeNet(self.in_features, self.num_classes, cfg)

        for layer, layer_ in zip(lenet.layers, self.layers):
            if isinstance(layer, nn.Linear) and isinstance(layer_, BayesianLinear):
                layer.weight.data = layer_.weight.data
                if layer_.bias is not None:
                    layer.bias.data = layer_.bias.data

        return lenet
