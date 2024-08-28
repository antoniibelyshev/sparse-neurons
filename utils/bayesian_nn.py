import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter

import math
from typing import Any
from functools import partial

from .utils import safe_sqrt, safe_log, safe_div


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
        self.weight_std = Parameter(Tensor(out_features, *conv_dims))

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
        nn.init.kaiming_normal_(self.weight)

        with torch.no_grad():
            self.weight_std.copy_(self.weight * 1e-4)
        
        if self.bias is not None:
            # std = math.sqrt(2 / math.prod(self.weight.shape[1:]))
            # nn.init.normal_(self.bias, 0, std)
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = self.linear_transform(x, self.weight, self.bias)
            sigma = safe_sqrt(self.linear_transform(x.pow(2), self.weight_std.pow(2)))
            return mu + torch.randn_like(sigma) * sigma
        return self.linear_transform(x, self.eval_weight, self.bias)

    def kl(self) -> Tensor:
        a = torch.ones(self.weight.shape[0], 1, device=self.device)
        b = torch.ones(1, self.weight.shape[1], device=self.device)

        s = self.weight.pow(2) + self.weight_std.pow(2)
        if s.dim() > 2:
            s = s.mean(list(range(2, s.dim())))

        for _ in range(3):
            a = (s / b).mean(1, keepdims=True)
            b = (s / a).mean(0, keepdims=True)

        kl = (safe_log(a).mean() + safe_log(b).mean()) * self.weight.numel()
        
        # kl = safe_log((self.weight.pow(2) + self.weight_std.pow(2)).mean(list(range(1, self.weight.dim())))).mean() * self.weight.numel()
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

    def get_mask(self, threshold: float | None = None) -> Tensor:
        return self.equivalent_dropout_rate < (threshold or self.threshold)

    @property
    def eval_weight(self) -> Tensor:
        return torch.where(self.get_mask(), self.weight, torch.zeros(1, device=self.device))
