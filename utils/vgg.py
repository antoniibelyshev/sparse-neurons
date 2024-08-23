import torch
from torch import nn, Tensor
from .bayesian_nn import BayesianLinear


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
    ) -> None:
        
        super(VGG, self).__init__() # type: ignore

        self.features_layers = nn.ModuleList()

        for cfg in features_cfg:
            if isinstance(cfg, int):
                self.features_layers.append(nn.Conv2d(
                    in_channels,
                    cfg,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                    stride=conv_stride,
                ))
                in_channels = cfg
                self.features_layers.append(nn.ReLU())
                self.features_layers.append(nn.BatchNorm2d(cfg))
                self.features_layers.append(nn.Dropout2d())
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
