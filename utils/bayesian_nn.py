import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter

from typing import Any
from functools import partial

from .utils import safe_sqrt, safe_log, safe_div
from .gaussian_mixture import em_gaussian_mixture, e_step, m_step


class BayesianLinear(nn.Module):
    sigma1: Tensor | None = None
    sigma2: Tensor | None = None
    p: Tensor | None = None

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
        """
        Initializes BayesianLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            conv_dims (tuple[int, ...] | None): Convolutional dimensions.
            bias (bool): Whether to include bias. Default is True.
            threshold (float): Dropout rate threshold. Default is 0.99.
            linear_transform_type (str): Type of linear transformation function.
            **linear_transform_kwargs (Any): Additional arguments for the linear transformation.
        """
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
        """
        Resets parameters to initial values.
        """
        nn.init.kaiming_normal_(self.weight)

        with torch.no_grad():
            self.weight_std.copy_(self.weight * 1e-4)
        
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.training:
            mu = self.linear_transform(x, self.weight, self.bias)
            sigma = safe_sqrt(self.linear_transform(x.pow(2), self.weight_std.pow(2)))
            return mu + torch.randn_like(sigma) * sigma
        return self.linear_transform(x, self.eval_weight, self.bias)

    def kl(self) -> Tensor:
        """
        Computes the KL divergence regularization term.

        Returns:
            Tensor: KL divergence.
        """
        x = safe_sqrt(self.weight.pow(2) + self.weight_std.pow(2))

        dims = list(range(1, x.dim()))
        if self.sigma1 is None or self.sigma2 is None or self.p is None:
            _, self.sigma1, self.sigma2, self.p = em_gaussian_mixture(x.detach(), dims, n_iter=10)

        pi = e_step(x, self.sigma1.detach(), self.sigma2.detach(), self.p.detach())
        self.sigma1, self.sigma2, self.p = m_step(x, pi, dims)

        x = self.weight + torch.randn_like(self.weight_std) * self.weight_std

        log_p1 = safe_log(self.p) - safe_log(self.sigma1) - 0.5 * safe_div(x, self.sigma1).pow(2)
        log_p2 = safe_log(1 - self.p) - safe_log(self.sigma2) - 0.5 * safe_div(x, self.sigma2).pow(2)

        m = torch.where(log_p1 > log_p2, log_p1, log_p2)

        log_p1 -= m
        log_p2 -= m

        kl = -m.sum() - safe_log(log_p1.exp() + log_p2.exp()).sum()
        kl -= 0.5 * (safe_log(self.weight_std.pow(2)).sum() + self.weight.numel())

        return kl

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the weight tensor.

        Returns:
            torch.device: Device of the weight tensor.
        """
        return self.weight.device

    @property
    def equivalent_dropout_rate(self) -> Tensor:
        """
        Computes the equivalent dropout rate.

        Returns:
            Tensor: Equivalent dropout rate.
        """
        alpha = self.weight.pow(2) / self.weight_std.pow(2)
        return 1 / (alpha + 1)

    def get_weight_mask(self, threshold: float | None = None) -> Tensor:
        """
        Computes a mask based on the dropout rate.

        Args:
            threshold (float | None): Dropout rate threshold. Uses class threshold if None.

        Returns:
            Tensor: Weight mask.
        """
        return self.equivalent_dropout_rate < (threshold or self.threshold)
    
    def get_in_mask(self) -> Tensor:
        """
        Computes the input mask.

        Returns:
            Tensor: Input mask.
        """
        return self.get_weight_mask().sum([0, *range(2, self.weight.dim())]) > 0
    
    def get_out_mask(self) -> Tensor:
        """
        Computes the output mask.

        Returns:
            Tensor: Output mask.
        """
        return self.get_weight_mask().sum(list(range(1, self.weight.dim()))) > 0

    @property
    def eval_weight(self) -> Tensor:
        """
        Returns the weight tensor used for evaluation.

        Returns:
            Tensor: Evaluation weight tensor.
        """
        return torch.where(self.get_weight_mask(), self.weight, torch.zeros(1, device=self.device))
