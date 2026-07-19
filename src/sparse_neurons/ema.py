"""Exponential moving averages of trainable parameters."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ParameterEMA:
    """Track an EMA of parameters while leaving analytical buffers untouched."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow: dict[str, Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            parameter.copy_(self.shadow[name])
