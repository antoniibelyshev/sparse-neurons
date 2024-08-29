import torch
from torch import nn, Tensor
from .bayesian_nn import BayesianLinear

import math


vgg_cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(
            self,
            cfg: str | list[int | str],
            in_channels: int = 3,
            num_classes: int = 10,
            *,
            conv_kernel_size: int = 3,
            conv_padding: int = 1,
            conv_stride: int = 1,
            max_pool_kernel_size: int = 2,
            max_pool_stride: int = 2,
            max_pool_padding: int = 0,
            bias: bool = True,
    ) -> None:
        
        super(VGG, self).__init__() # type: ignore

        self.features_layers = nn.ModuleList()

        if isinstance(cfg, str):
            cfg = vgg_cfgs[cfg]

        for i, cfg_ in enumerate(cfg):
            if isinstance(cfg_, int):
                self.features_layers.append(nn.Conv2d(
                    in_channels,
                    cfg_,
                    bias=bias,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                    stride=conv_stride,
                ))

                std = math.sqrt(2 / conv_kernel_size ** 2 / cfg_)
                nn.init.normal_(self.features_layers[-1].weight, 0, std)
                if bias:
                    nn.init.zeros_(self.features_layers[-1].bias)

                in_channels = cfg_

                self.features_layers.append(nn.ReLU())
                self.features_layers.append(nn.BatchNorm2d(cfg_, eps=1e-3))
                self.features_layers.append(nn.Dropout2d(0.4 if i else 0.3))
            elif cfg_ == 'M':
                self.features_layers.pop(-1)
                self.features_layers.append(nn.MaxPool2d(
                    kernel_size=max_pool_kernel_size,
                    stride=max_pool_stride,
                    padding=max_pool_padding,
                ))
            else:
                raise ValueError(f'Unsupported layer type: {cfg_}')

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.features_layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_conv_layers(self) -> list[nn.Conv2d]:
        return [layer for layer in self.features_layers if isinstance(layer, nn.Conv2d)]

    def get_fc_layers(self) -> list[nn.Linear]:
        return [layer for layer in self.classifier if isinstance(layer, nn.Linear)]


class BayesianVGG(nn.Module):
    def __init__(
        self,
        cfg: list[int | str],
        in_channels: int = 3,
        num_classes: int = 10,
        *,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        conv_stride: int = 1,
        max_pool_kernel_size: int = 2,
        max_pool_stride: int = 2,
        max_pool_padding: int = 0,
        threshold: float = 0.99,
    ) -> None:

        super(BayesianVGG, self).__init__()  # type: ignore

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.ModuleList()

        if isinstance(cfg, str):
            cfg = vgg_cfgs[cfg]

        for cfg_ in cfg:
            if isinstance(cfg_, int):
                self.features.append(BayesianLinear(
                    in_channels,
                    cfg_,
                    (conv_kernel_size, conv_kernel_size),
                    linear_transform_type='conv2d',
                    stride=conv_stride,
                    padding=conv_padding,
                    threshold=threshold,
                ))
                in_channels = cfg_

                self.features.append(nn.ReLU())
                self.features.append(nn.BatchNorm2d(cfg_))
            elif cfg_ == 'M':
                self.features.append(nn.MaxPool2d(
                    kernel_size=max_pool_kernel_size,
                    stride=max_pool_stride,
                    padding=max_pool_padding,
                ))
            else:
                raise ValueError(f'Unsupported layer type: {cfg_}')

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.features:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def kl(self) -> Tensor:
        kl = torch.zeros(1, device=self.device)
        for layer in self.features:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl()
        for layer in self.classifier:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl()
            if isinstance(layer, nn.Linear):
                kl += layer.weight.pow(2).sum() + layer.bias.pow(2).sum()
        return kl

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_conv_layers(self) -> list[BayesianLinear]:
        return [layer for layer in self.features if isinstance(layer, BayesianLinear)]

    def get_fc_layers(self) -> list[nn.Linear]:
        return [layer for layer in self.classifier if isinstance(layer, nn.Linear)]
    
    def get_bayesian_layers(self) -> list[BayesianLinear]:
        return self.get_conv_layers()

    def from_pretrained(self, vgg: VGG) -> None:
        for layer, layer_ in zip(self.get_conv_layers(), vgg.get_conv_layers()):
            layer.weight.data = layer_.weight.data
            layer.weight_std.data = layer_.weight.data * 1e-4

            if layer.bias is not None and layer_.bias is not None:
                layer.bias.data = layer_.bias.data

        self.classifier = vgg.classifier

    def squeezed_cfg(self) -> list[int | str]:
        cfg: list[int | str] = []
        for layer in self.features:
            if isinstance(layer, BayesianLinear):
                cfg.append(int(layer.get_out_mask().sum()))
            if isinstance(layer, nn.MaxPool2d):
                cfg.append('M')
        return cfg

    def squeeze(self) -> VGG:
        vgg = VGG(self.squeezed_cfg(), self.in_channels, self.num_classes)

        in_mask = torch.ones(self.get_conv_layers()[0].weight.shape[1])
        for layer, layer_ in zip(vgg.get_conv_layers(), self.get_conv_layers()):
            out_mask = layer_.get_out_mask()
            layer.weight.data = layer_.weight.data[out_mask, in_mask]

            if layer.bias is not None and layer_.bias is not None:
                layer.bias.data = layer_.bias.data[out_mask]

        return vgg
