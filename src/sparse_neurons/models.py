"""Reference architectures using structured variational layers."""

from __future__ import annotations

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
