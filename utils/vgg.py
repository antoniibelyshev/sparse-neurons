import torch
from torch import nn, Tensor
from .bayesian_nn import BayesianLinear

from .utils import safe_log
import math


class VGG(nn.Module):
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
            bias: bool = True,
    ) -> None:
        
        super(VGG, self).__init__() # type: ignore

        self.features_layers = nn.ModuleList()

        for i, cfg in enumerate(features_cfg):
            if isinstance(cfg, int):
                self.features_layers.append(nn.Conv2d(
                    in_channels,
                    cfg,
                    bias=bias,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                    stride=conv_stride,
                ))

                std = math.sqrt(2 / conv_kernel_size ** 2 / cfg)
                nn.init.normal_(self.features_layers[-1].weight, 0, std)
                if bias:
                    nn.init.zeros_(self.features_layers[-1].bias)

                in_channels = cfg

                self.features_layers.append(nn.ReLU())
                self.features_layers.append(nn.BatchNorm2d(cfg, eps=1e-3))
                self.features_layers.append(nn.Dropout2d(0.4 if i else 0.3))
                # self.features_layers.append(nn.Dropout2d(0.5))
            elif cfg == 'M':
                self.features_layers.pop(-1)
                self.features_layers.append(nn.MaxPool2d(
                    kernel_size=max_pool_kernel_size,
                    stride=max_pool_stride,
                    padding=max_pool_padding,
                ))
            else:
                raise ValueError(f'Unsupported layer type: {cfg}')
            
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

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
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
    ) -> None:

        super(BayesianVGG, self).__init__()  # type: ignore

        self.features = nn.ModuleList()

        for cfg in features_cfg:
            if isinstance(cfg, int):
                self.features.append(BayesianLinear(
                    (in_channels, conv_kernel_size, conv_kernel_size),
                    cfg,
                    linear_transform_type='conv2d',
                    stride=conv_stride,
                    padding=conv_padding
                ))
                in_channels = cfg

                self.features.append(nn.ReLU())
                self.features.append(nn.BatchNorm2d(cfg))
            elif cfg == 'M':
                self.features.append(nn.MaxPool2d(
                    kernel_size=max_pool_kernel_size,
                    stride=max_pool_stride,
                    padding=max_pool_padding,
                ))
            else:
                raise ValueError(f'Unsupported layer type: {cfg}')
            
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # self.classifier = nn.Sequential(
        #     BayesianLinear(512, 512, linear_transform_type='linear'),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(512),
        #     BayesianLinear(512, 512, linear_transform_type='linear'),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(512),
        #     BayesianLinear(512, num_classes, linear_transform_type='linear'),
        # )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.features:
            x = layer(x)
        # x = self.avgpool(x)
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

    def squeeze_features(self, threshold: float | None = None) -> nn.Module:
        squeezed_features: list[nn.Module] = []

        in_mask = None
        for layer in self.features:
            if isinstance(layer, BayesianLinear):
                layer = layer.squeeze(in_mask, threshold)
                in_mask = layer.get_mask(threshold)
                squeezed_features.append(layer)
            else:
                squeezed_features.append(layer)

        return nn.Sequential(*squeezed_features)

    # def squeeze_classifier(self, threshold: float | None = None) -> nn.Module:
    #     squeezed_classifier: list[nn.Module] = []

    #     in_mask = self.last_conv2d.get_mask(threshold).reshape(-1, 1, 1).repeat(1, 7, 7).flatten(1)
    #     for layer in self.classifier:
    #         if isinstance(layer, BayesianLinear):
    #             layer = layer.squeeze(in_mask, threshold)
    #             in_mask = layer.get_mask(threshold)
    #             squeezed_classifier.append(layer)
    #         else:
    #             squeezed_classifier.append(layer)

    #     return nn.Sequential(*squeezed_classifier)

    # def squeeze(self, threshold: float | None = None) -> nn.Module:
    #     return nn.Sequential(self.squeeze_features(threshold), nn.Flatten(), self.squeeze_classifier(threshold))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def get_conv2d_bayesian(self):
        return [layer for layer in self.features if isinstance(layer, BayesianLinear)]

    def get_fc_bayesian(self):
        return [layer for layer in self.classifier if isinstance(layer, BayesianLinear)]

    def get_bayesian_layers(self) -> list[BayesianLinear]:
        return self.get_conv2d_bayesian() + self.get_fc_bayesian()

    def from_vgg(self, vgg: VGG) -> None:
        for layer, layer_ in zip(self.features, [layer for layer in vgg.features_layers if not isinstance(layer, nn.Dropout2d)]):
            if isinstance(layer, BayesianLinear):
                layer.weight.data = layer_.weight.data
                layer.weight_std.data = layer_.weight.data * 1e-4

                if layer.bias is not None:
                    layer.bias.data = layer_.bias.data

            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data = layer_.weight.data.clone()
                layer.bias.data = layer_.bias.data.clone()
                
                layer.running_mean.data = layer_.running_mean.data.clone()
                layer.running_var.data = layer_.running_var.data.clone()

                layer.momentum = layer_.momentum
                layer.eps = layer_.eps

        # for layer, layer_ in zip(self.classifier, [layer for layer in vgg.classifier if not isinstance(layer, nn.Dropout)]):
        #     if isinstance(layer, BayesianLinear):
        #         layer.weight.data = layer_.weight.data
        #         layer.weight_std.data = layer_.weight * 1e-1

        #         if layer.bias is not None:
        #             layer.bias.data = layer_.bias.data

        #         layer.update_ab()

        #     if isinstance(layer, nn.BatchNorm1d):
        #         layer.weight.data = layer_.weight.data.clone()
        #         layer.bias.data = layer_.bias.data.clone()
                
        #         layer.running_mean.data = layer_.running_mean.data.clone()
        #         layer.running_var.data = layer_.running_var.data.clone()

        #         layer.momentum = layer_.momentum
        #         layer.eps = layer_.eps

        self.classifier = vgg.classifier
