import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from typing import Any
from functools import partial

from .utils import safe_sqrt, safe_log


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BayesianLinear(nn.Module):
    def __init__(
        self,
        conv_dims: int | tuple[int, ...],
        out_features: int,
        bias: bool = True,
        threshold: float = 0.99,
        *,
        linear_transform_type: str,
        **linear_transform_kwargs: Any
    ):

        super(BayesianLinear, self).__init__()  # type: ignore

        if isinstance(conv_dims, int):
            conv_dims = (conv_dims,)

        self.weight = Parameter(Tensor(out_features, *conv_dims))
        self.weight_log_var = Parameter(Tensor(out_features, *conv_dims))

        self.bias = Parameter(Tensor(out_features)) if bias else None

        assert linear_transform_type in {
            'linear', 'conv2d'}, 'Unsupported linear_transform type'

        self.linear_transform_type = linear_transform_type
        self.linear_transform_kwargs = linear_transform_kwargs

        self.linear_transform = partial(
            getattr(nn.functional, linear_transform_type), **linear_transform_kwargs)

        self.threshold = threshold

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

        with torch.no_grad():
            self.weight_log_var.copy_(safe_log(self.weight ** 2) - 5)
        
        if self.bias is not None:
            bound = 1 / math.sqrt(self.conv_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = self.linear_transform(x, self.weight, self.bias)
            sigma = safe_sqrt(self.linear_transform(x.pow(2), self.weight_var))
            return mu + torch.randn_like(sigma) * sigma
        return self.linear_transform(x, self.eval_weight, self.bias)

    def kl(self) -> Tensor:
        kl = safe_log((self.weight.pow(
            2) + self.weight_var).mean(list(range(1, self.weight.dim())))).sum() * self.conv_size
        kl -= self.weight_log_var.sum()
        return kl / 2

    def squeeze(self, in_mask: Tensor | None, threshold: float | None = None) -> nn.Module:
        in_mask = in_mask or torch.ones(self.conv_size, device=self.device)
        out_mask = self.get_mask(threshold)
        layer_type = getattr(nn, self.linear_transform_type.capitalize())
        in_features = int(in_mask.sum())
        out_features = int(out_mask.sum())
        bias = self.bias is not None
        layer = layer_type(in_features, out_features,
                           bias=bias, **self.linear_transform_kwargs)
        layer.weight.data = self.weight[out_mask, in_mask]
        if bias:
            layer.bias.data = self.bias
        return layer

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def weight_var(self) -> Tensor:
        return self.weight_log_var.exp()

    @property
    def out_dim(self) -> int:
        return self.weight.shape[0]

    @property
    def conv_size(self) -> int:
        return math.prod(self.weight.shape[1:])

    @property
    def equivalent_dropout_rate(self) -> Tensor:
        alpha = self.weight.pow(2) / self.weight_var
        return 1 / (alpha + 1)

    def get_mask(self, threshold: float | None = None) -> Tensor:
        return self.equivalent_dropout_rate < (threshold or self.threshold)

    @property
    def eval_weight(self) -> Tensor:
        return torch.where(self.get_mask(), self.weight, torch.zeros(1, device=self.device))


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


class BayesianVGG(nn.Module):
    def __init__(
        self,
        features_cfg: list[int | str],
        in_channels: int = 3,
        num_classes: int = 10,
        *,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        conv_stride: int = 1,
        max_pool_kernel_size: int = 2,
        max_pool_stride: int = 2,
        max_pool_padding: int = 0,
    ):

        super(BayesianVGG, self).__init__()  # type: ignore

        self.features_layers = nn.ModuleList()

        for cfg in features_cfg:
            if isinstance(cfg, int):
                self.last_conv2d = BayesianLinear(
                    (in_channels, conv_kernel_size, conv_kernel_size),
                    cfg,
                    linear_transform_type='conv2d',
                    stride=conv_stride,
                    padding=conv_padding
                )
                self.features_layers.append(self.last_conv2d)
                in_channels = cfg

                self.features_layers.append(nn.ReLU())
            elif cfg == 'M':
                self.features_layers.append(nn.MaxPool2d(
                    kernel_size=max_pool_kernel_size,
                    stride=max_pool_stride,
                    padding=max_pool_padding,
                ))
            else:
                raise ValueError(f'Unsupported layer type: {cfg}')
            
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.features_layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def kl(self) -> Tensor:
        kl = torch.zeros(1, device=self.device)
        for layer in self.features_layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl()
        for layer in self.classifier:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl()
        return kl

    def squeeze_features(self, threshold: float | None = None) -> nn.Module:
        squeezed_features: list[nn.Module] = []

        in_mask = None
        for layer in self.features_layers:
            if isinstance(layer, BayesianLinear):
                layer = layer.squeeze(in_mask, threshold)
                in_mask = layer.get_mask(threshold)
                squeezed_features.append(layer)
            else:
                squeezed_features.append(layer)

        return nn.Sequential(*squeezed_features)

    def squeeze_classifier(self, threshold: float | None = None) -> nn.Module:
        squeezed_classifier: list[nn.Module] = []

        in_mask = self.last_conv2d.get_mask(threshold).reshape(-1, 1, 1).repeat(1, 7, 7).flatten(1)
        for layer in self.classifier:
            if isinstance(layer, BayesianLinear):
                layer = layer.squeeze(in_mask, threshold)
                in_mask = layer.get_mask(threshold)
                squeezed_classifier.append(layer)
            else:
                squeezed_classifier.append(layer)

        return nn.Sequential(*squeezed_classifier)

    def squeeze(self, threshold: float | None = None) -> nn.Module:
        return nn.Sequential(self.squeeze_features(threshold), self.avgpool, nn.Flatten(), self.squeeze_classifier(threshold))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_bayesian_layers(self) -> list[BayesianLinear]:
        return [layer for layer in self.features_layers if isinstance(layer, BayesianLinear)] + [layer for layer in self.classifier if isinstance(layer, BayesianLinear)]
