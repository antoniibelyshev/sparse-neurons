import torch
from torch import nn, Tensor

from .bayesian_nn import BayesianLinear


class LeNet(nn.Module):
    def __init__(
            self,
            in_features: int = 728,
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
            in_features: int = 728,
            num_classes: int = 10,
            intermediate_dims: list[int] = [500, 300],
            bias: bool = True,
            threshold: float = 0.99,
    ):
        super(BayesianLeNet, self).__init__() # type: ignore

        self.layers = nn.ModuleList()

        for out_features in intermediate_dims:
            self.layers.append(BayesianLinear(in_features, out_features, bias, threshold, linear_transform_type='linear'))
            self.layers.append(nn.ReLU())
            in_features = out_features

        self.layers.append(nn.Linear(in_features, num_classes, bias))

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
    
    def squeeze(self, threshold: float | None = None) -> nn.Module:
        squeezed_layers: list[nn.Module] = []

        in_mask = None
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                layer = layer.squeeze(in_mask, threshold)
                in_mask = layer.get_mask(threshold)
                squeezed_layers.append(layer)
            else:
                squeezed_layers.append(layer)

        return nn.Sequential(*squeezed_layers)
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_bayesian_layers(self) -> list[BayesianLinear]:
        return [layer for layer in self.layers if isinstance(layer, BayesianLinear)]
