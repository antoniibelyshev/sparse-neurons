"""Reference architectures using structured variational layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sparse_neurons.conversion import ard_kl_divergence, update_log_lambdas
from sparse_neurons.layers import GroupARDLinear


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


class LeNet300100(nn.Module):
    """The fully connected 784-300-100-10 MNIST architecture."""

    def __init__(
        self,
        *,
        initial_log_variance: float = -12.0,
        hidden_sizes: tuple[int, int] = (300, 100),
    ) -> None:
        super().__init__()
        hidden_1, hidden_2 = hidden_sizes
        self.layers = nn.ModuleList(
            [
                GroupARDLinear(784, hidden_1, initial_log_variance=initial_log_variance),
                GroupARDLinear(hidden_1, hidden_2, initial_log_variance=initial_log_variance),
                GroupARDLinear(hidden_2, 10, initial_log_variance=initial_log_variance),
            ]
        )

    def forward(self, x: Tensor, *, sample: bool | None = None) -> Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(self.layers[0](x, sample=sample))
        x = torch.relu(self.layers[1](x, sample=sample))
        return self.layers[2](x, sample=sample)

    def update_log_lambdas(self) -> None:
        update_log_lambdas(self)

    def kl_divergence(self) -> Tensor:
        return ard_kl_divergence(self)
