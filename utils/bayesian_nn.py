import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter

from typing import Any
from functools import partial

import numpy as np

from .utils import safe_sqrt, safe_log, t_log_likelihood


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv_dims: tuple[int, ...] | None = None,
        *,
        bias: bool = True,
        threshold: float = 0.99,
        linear_transform_type: str,
        **linear_transform_kwargs: Any
    ):
        super(BayesianLinear, self).__init__()  # type: ignore

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor(out_features, in_features, *(conv_dims or ())))
        self.weight_std = Parameter(Tensor(out_features, in_features, *(conv_dims or ())))

        self.bias = Parameter(Tensor(out_features)) if bias else None

        self.linear_transform_type = linear_transform_type
        self.linear_transform_kwargs = linear_transform_kwargs

        self.linear_transform = partial(getattr(nn.functional, linear_transform_type), **linear_transform_kwargs)

        self.threshold = threshold

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

        with torch.no_grad():
            self.weight_std.copy_(self.weight * 1e-4)
        
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = self.linear_transform(x, self.weight, self.bias)
            sigma = safe_sqrt(self.linear_transform(x.pow(2), self.weight_std.pow(2)))
            return mu + torch.randn_like(sigma) * sigma
        return self.linear_transform(x, self.eval_weight, self.bias)

    def kl(self) -> Tensor:
        s = self.weight.pow(2) + self.weight_std.pow(2)
        if s.dim() > 2:
            s = s.mean(list(range(2, s.dim())))

        a = torch.ones(self.out_features, 1, device=self.device)
        b = torch.ones(1, self.in_features, device=self.device)

        for _ in range(3):
            a = (s / b).mean(1, keepdims=True) # type: ignore
            b = (s / a).mean(0, keepdims=True) # type: ignore

        kl = (safe_log(a).mean() + safe_log(b).mean()) * self.weight.numel() # type: ignore
        kl -= safe_log(self.weight_std.pow(2)).sum()

        if self.bias is not None:
            kl += self.bias.pow(2).sum()

        return kl / 2

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def equivalent_dropout_rate(self) -> Tensor:
        alpha = self.weight.pow(2) / self.weight_std.pow(2)
        return 1 / (alpha + 1)

    def get_weight_mask(self, threshold: float | None = None) -> Tensor:
        return self.equivalent_dropout_rate < (threshold or self.threshold)
    
    def get_in_mask(self) -> Tensor:
        return self.get_weight_mask().any([0, *range(2, self.weight.dim())]) > 0
    
    def get_out_mask(self) -> Tensor:
        return self.get_weight_mask().any(list(range(1, self.weight.dim()))) > 0

    @property
    def eval_weight(self) -> Tensor:
        return torch.where(self.get_weight_mask(), self.weight, torch.zeros(1, device=self.device))


class TDistributedBayesianLinear(BayesianLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv_dims: tuple[int, ...] | None = None,
        **kwargs: Any,
    ):
        super(TDistributedBayesianLinear, self).__init__(in_features, out_features, conv_dims, **kwargs)  # type: ignore

        self.register_buffer("nu", torch.ones(out_features))

        self.reset_parameters()

    def kl(self) -> Tensor:
        samples = self.weight + torch.randn_like(self.weight_std) * self.weight_std
        kl = t_log_likelihood(samples, self.nu)
        kl -= 0.5 * (safe_log(self.weight_std.pow(2)).sum() + (np.log(2 * np.pi) + 1) * self.weight.numel())
        return kl / 2
