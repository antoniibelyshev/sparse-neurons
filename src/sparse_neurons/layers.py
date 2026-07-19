"""Variational layers with group automatic relevance determination."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TwoSidedGroupARDLinear(nn.Module):
    """Linear layer with independent input and output ARD scale vectors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        initial_log_variance: float = -12.0,
        variance_floor: float = 1e-12,
        m_step_sweeps: int = 2,
        mixture_spike_variance: float = 1e-4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.variance_floor = variance_floor
        self.m_step_sweeps = m_step_sweeps
        if mixture_spike_variance <= 0.0:
            raise ValueError("mixture_spike_variance must be positive")
        self.mixture_spike_variance = mixture_spike_variance
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_variance = nn.Parameter(
            torch.full((out_features, in_features), initial_log_variance)
        )
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_log_variance = nn.Parameter(
                torch.full((out_features,), initial_log_variance)
            )
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_log_variance", None)
        self.augmented_in_features = in_features + int(bias)
        self.register_buffer("log_lambda_in", torch.zeros(self.augmented_in_features))
        self.register_buffer("log_lambda_out", torch.zeros(out_features))
        self.register_buffer("spike_probability", torch.tensor(0.5))
        self.register_buffer(
            "log_spike_variance",
            torch.tensor(math.log(mixture_spike_variance)),
        )
        self.register_buffer(
            "spike_responsibility",
            torch.zeros(out_features, self.augmented_in_features),
        )
        self.reset_parameters()

    @property
    def log_lambda(self) -> Tensor:
        """Output precision alias used by common row diagnostics."""
        return self.log_lambda_out

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias_mu is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias_mu, -bound, bound)
        self.update_log_lambda()

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        initial_log_variance: float = -12.0,
        variance_floor: float = 1e-12,
        m_step_sweeps: int = 2,
        initial_relative_variance: float | None = None,
        mixture_spike_variance: float = 1e-4,
    ) -> TwoSidedGroupARDLinear:
        converted = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            initial_log_variance=initial_log_variance,
            variance_floor=variance_floor,
            m_step_sweeps=m_step_sweeps,
            mixture_spike_variance=mixture_spike_variance,
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            converted.weight_mu.copy_(layer.weight)
            if layer.bias is not None:
                converted.bias_mu.copy_(layer.bias)
            if initial_relative_variance is not None:
                converted.weight_log_variance.copy_(
                    (
                        variance_floor
                        + initial_relative_variance * converted.weight_mu.square()
                    ).log()
                )
                if converted.bias_mu is not None:
                    converted.bias_log_variance.copy_(
                        (
                            variance_floor
                            + initial_relative_variance * converted.bias_mu.square()
                        ).log()
                    )
            converted.update_log_lambda()
        converted.train(layer.training)
        return converted

    def weight_second_moment(self) -> Tensor:
        return self.weight_mu.square() + self.weight_log_variance.exp()

    def augmented_weight_mu(self) -> Tensor:
        if self.bias_mu is None:
            return self.weight_mu
        return torch.cat((self.weight_mu, self.bias_mu[:, None]), dim=1)

    def augmented_weight_log_variance(self) -> Tensor:
        if self.bias_log_variance is None:
            return self.weight_log_variance
        return torch.cat(
            (self.weight_log_variance, self.bias_log_variance[:, None]), dim=1
        )

    def augmented_weight_second_moment(self) -> Tensor:
        mean = self.augmented_weight_mu()
        return mean.square() + self.augmented_weight_log_variance().exp()

    def row_energy(self) -> Tensor:
        return self.augmented_weight_second_moment().sum(1)

    @torch.no_grad()
    def update_log_lambda(self) -> Tensor:
        second_moment = self.augmented_weight_second_moment()
        for _ in range(self.m_step_sweeps):
            slab_responsibility = self._update_mixture(second_moment)
            output_energy = (
                second_moment
                * slab_responsibility
                * self.log_lambda_in.exp()[None, :]
            ).sum(1)
            output_dimension = slab_responsibility.sum(1)
            self.log_lambda_out.copy_(
                output_dimension.clamp_min(self.variance_floor).log()
                - output_energy.clamp_min(self.variance_floor).log()
            )
            input_energy = (
                second_moment
                * slab_responsibility
                * self.log_lambda_out.exp()[:, None]
            ).sum(0)
            input_dimension = slab_responsibility.sum(0)
            updated_log_lambda_in = (
                input_dimension.clamp_min(self.variance_floor).log()
                - input_energy.clamp_min(self.variance_floor).log()
            )
            self.log_lambda_in.copy_(updated_log_lambda_in)
        return self.log_lambda_out

    @torch.no_grad()
    def _update_mixture(self, second_moment: Tensor) -> Tensor:
        base_precision = (
            self.log_lambda_out.exp()[:, None]
            * self.log_lambda_in.exp()[None, :]
        )
        probability = self.spike_probability.clamp(1e-6, 1.0 - 1e-6)
        logit_probability = probability.log() - (-probability).log1p()
        spike_precision = (-self.log_spike_variance).exp()
        responsibility_logit = (
            logit_probability
            - 0.5 * (base_precision.log() + self.log_spike_variance)
            - 0.5 * second_moment * (spike_precision - base_precision)
        )
        self.spike_responsibility.copy_(responsibility_logit.sigmoid())
        self.spike_probability.copy_(
            self.spike_responsibility.mean().clamp(1e-6, 1.0 - 1e-6)
        )
        spike_mass = self.spike_responsibility.sum()
        spike_energy = (self.spike_responsibility * second_moment).sum()
        spike_variance = (
            spike_energy / spike_mass.clamp_min(self.variance_floor)
        ).clamp_min(self.variance_floor)
        self.log_spike_variance.copy_(spike_variance.log())
        return 1.0 - self.spike_responsibility

    def kl_divergence(self) -> Tensor:
        base_precision = (
            self.log_lambda_out[:, None].exp()
            * self.log_lambda_in[None, :].exp()
        )
        second_moment = self.augmented_weight_second_moment()
        log_variance = self.augmented_weight_log_variance()
        responsibility = self.spike_responsibility.clamp(1e-6, 1.0 - 1e-6)
        slab_responsibility = 1.0 - responsibility
        probability = self.spike_probability.clamp(1e-6, 1.0 - 1e-6)
        spike_precision = (-self.log_spike_variance).exp()
        gaussian_kl = 0.5 * (
            (responsibility * spike_precision + slab_responsibility * base_precision)
            * second_moment
            - 1.0
            - log_variance
            + responsibility * self.log_spike_variance
            - slab_responsibility
            * (self.log_lambda_out[:, None] + self.log_lambda_in[None, :])
        )
        categorical_kl = (
            responsibility * (responsibility.log() - probability.log())
            + slab_responsibility
            * (slab_responsibility.log() - (1.0 - probability).log())
        )
        edge_kl = gaussian_kl + categorical_kl
        return edge_kl.sum()

    def forward(self, input: Tensor, *, sample: bool | None = None) -> Tensor:
        if sample is None:
            sample = self.training
        mean = F.linear(input, self.weight_mu, self.bias_mu)
        if not sample:
            return mean
        variance = F.linear(
            input.square(),
            self.weight_log_variance.exp(),
            None if self.bias_log_variance is None else self.bias_log_variance.exp(),
        )
        return mean + (variance + self.variance_floor).sqrt() * torch.randn_like(mean)
