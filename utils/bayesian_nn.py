import torch
from torch import nn, Tensor
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

        self.a = Parameter(torch.ones(out_features, 1), requires_grad=True)
        self.b = Parameter(torch.ones(1, conv_dims[0]), requires_grad=True)

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
            self.weight_log_var.copy_(safe_log(self.weight ** 2) - 5)
        
        if self.bias is not None:
            std = math.sqrt(2 / self.conv_size)
            nn.init.normal_(self.bias, 0, std)

        self.update_ab()

    def update_ab(self, n_iter: int = 10):
        s = self.weight.pow(2) + self.weight_var
        if s.dim() > 2:
            s = s.mean(list(range(2, self.weight.dim())))
        for _ in range(n_iter):
            self.a.data = (s / self.b).mean(1, keepdims=True)
            self.b.data = (s / self.a).mean(0, keepdims=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = self.linear_transform(x, self.weight, self.bias)
            sigma = safe_sqrt(self.linear_transform(x.pow(2), self.weight_var))
            return mu + torch.randn_like(sigma) * sigma
        return self.linear_transform(x, self.eval_weight, self.bias)

    def kl(self) -> Tensor:
        # kl = safe_log((self.weight.pow(
        #     2) + self.weight_var).mean(list(range(1, self.weight.dim())))).sum() * self.conv_size
        # kl -= self.weight_log_var.sum()
        # return kl / 2
        # s = self.weight.pow(2) + self.weight_var
        
        # with torch.no_grad():
        #     self.a.copy_((s / self.b).mean([1, *range(2, s.dim())], keepdims=True))
        #     self.b.copy_((s / self.a).mean([0, *range(2, s.dim())], keepdims=True))

        # sigma = (self.a * self.b).detach()
        # return ((s.sum(list(range(2, s.dim()))) / sigma).sum() + safe_log(sigma).sum() - self.weight_log_var.sum() - math.prod(s.shape)) / 2

        s = self.weight.pow(2) + self.weight_var
        if s.dim() > 2:
            s = s.sum(list(range(2, s.dim())))
        kl = safe_log(self.a * self.b).sum() * math.prod(self.weight.shape[2:]) - self.weight_log_var.sum()
        kl += (s / (self.a * self.b)).sum() - math.prod(self.weight.shape)
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
