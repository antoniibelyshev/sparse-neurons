"""Reference architectures using structured variational layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeterministicLeNet300100(nn.Module):
    """Standard deterministic 784-300-100-10 MNIST network."""

    def __init__(self, hidden_sizes: tuple[int, int] = (300, 100)) -> None:
        super().__init__()
        hidden_1, hidden_2 = hidden_sizes
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DeterministicVGG11Cifar10(nn.Module):
    """VGG-11 adapted to 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        configuration: list[int | str] = [
            64,
            "M",
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ]
        layers: list[nn.Module] = []
        in_channels = 3
        for item in configuration:
            if item == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                out_channels = int(item)
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))
