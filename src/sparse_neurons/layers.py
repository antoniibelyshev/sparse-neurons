"""Variational layers with group automatic relevance determination."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GroupARDLinear(nn.Module):
    """Linear layer with one empirical-Bayes precision per output neuron.

    Every scalar weight has a diagonal Gaussian variational posterior. The
    incoming weights and bias of one output neuron form a group and share the
    prior precision ``lambda``. The precision is stored as ``log_lambda`` and
    updated to its exact M-step optimum rather than trained by gradient descent.

    Args:
        in_features: Number of input features.
        out_features: Number of output features (and row groups).
        bias: Include a bias in both the affine transform and its row group.
        initial_log_variance: Initial posterior log variance for every scalar.
        a0: Optional Gamma-prior shape. Set both ``a0`` and ``b0`` to ``None``
            for maximum-likelihood precisions ``lambda = d / S``.
        b0: Optional Gamma-prior rate.
        variance_floor: Numerical floor used in sampling and logarithms.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        initial_log_variance: float = -12.0,
        a0: float | None = None,
        b0: float | None = None,
        variance_floor: float = 1e-12,
    ) -> None:
        super().__init__()
        if (a0 is None) != (b0 is None):
            raise ValueError("a0 and b0 must either both be set or both be None")
        if a0 is not None and (a0 <= 0.0 or b0 < 0.0):
            raise ValueError("Gamma hyperparameters require a0 > 0 and b0 >= 0")

        self.in_features = in_features
        self.out_features = out_features
        self.a0 = a0
        self.b0 = b0
        self.variance_floor = variance_floor

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

        self.register_buffer("log_lambda", torch.zeros(out_features))
        self.reset_parameters()

    @property
    def group_size(self) -> int:
        """Number of random scalars in each row group."""
        return self.in_features + int(self.bias_mu is not None)

    def reset_parameters(self) -> None:
        """Match ``torch.nn.Linear`` initialization for posterior means."""
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
        a0: float | None = None,
        b0: float | None = None,
        variance_floor: float = 1e-12,
        initial_relative_variance: float | None = None,
    ) -> GroupARDLinear:
        """Create a variational layer whose posterior mean copies ``layer``."""
        converted = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            initial_log_variance=initial_log_variance,
            a0=a0,
            b0=b0,
            variance_floor=variance_floor,
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

    def row_energy(self) -> Tensor:
        """Return ``E_q[||w_i||^2]`` for every augmented row."""
        energy = (self.weight_mu.square() + self.weight_log_variance.exp()).sum(1)
        if self.bias_mu is not None:
            energy = energy + self.bias_mu.square() + self.bias_log_variance.exp()
        return energy

    @torch.no_grad()
    def update_log_lambda(self) -> Tensor:
        """Perform the exact empirical-Bayes/MAP M-step in log space."""
        energy = self.row_energy().clamp_min(self.variance_floor)
        d = float(self.group_size)
        if self.a0 is None:
            new_log_lambda = math.log(d) - energy.log()
        else:
            numerator = self.a0 + d / 2.0 - 1.0
            if numerator <= 0.0:
                raise ValueError("MAP update requires a0 + group_size / 2 > 1")
            denominator = self.b0 + energy / 2.0
            new_log_lambda = math.log(numerator) - denominator.log()
        self.log_lambda.copy_(new_log_lambda)
        return self.log_lambda

    def kl_divergence(self) -> Tensor:
        """Return ``KL[q(W) || p(W | lambda)]`` summed over all rows."""
        energy = self.row_energy()
        log_determinant = self.weight_log_variance.sum(1)
        if self.bias_log_variance is not None:
            log_determinant = log_determinant + self.bias_log_variance
        d = float(self.group_size)
        row_kl = 0.5 * (
            self.log_lambda.exp() * energy
            - d
            - d * self.log_lambda
            - log_determinant
        )
        return row_kl.sum()

    def forward(self, input: Tensor, *, sample: bool | None = None) -> Tensor:
        """Apply the layer, sampling preactivations during training by default."""
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

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias_mu is not None}, group_size={self.group_size}"
        )


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
        fix_output_scale: bool = False,
        mixture_spike_variance: float | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.variance_floor = variance_floor
        self.m_step_sweeps = m_step_sweeps
        self.fix_output_scale = fix_output_scale
        if mixture_spike_variance is not None and mixture_spike_variance <= 0.0:
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
        self.register_buffer("log_lambda_in", torch.zeros(in_features))
        self.register_buffer("log_lambda_out", torch.zeros(out_features))
        self.register_buffer("spike_probability", torch.tensor(0.5))
        self.register_buffer(
            "log_spike_variance",
            torch.tensor(
                0.0 if mixture_spike_variance is None else math.log(mixture_spike_variance)
            ),
        )
        self.register_buffer(
            "spike_responsibility", torch.zeros(out_features, in_features)
        )
        self.register_buffer("bias_spike_responsibility", torch.zeros(out_features))
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
        fix_output_scale: bool = False,
        initial_relative_variance: float | None = None,
        mixture_spike_variance: float | None = None,
    ) -> TwoSidedGroupARDLinear:
        converted = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            initial_log_variance=initial_log_variance,
            variance_floor=variance_floor,
            m_step_sweeps=m_step_sweeps,
            fix_output_scale=fix_output_scale,
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

    def row_energy(self) -> Tensor:
        energy = self.weight_second_moment().sum(1)
        if self.bias_mu is not None:
            energy = energy + self.bias_mu.square() + self.bias_log_variance.exp()
        return energy

    @torch.no_grad()
    def update_log_lambda(self) -> Tensor:
        second_moment = self.weight_second_moment()
        for _ in range(self.m_step_sweeps):
            precision_multiplier = self._update_mixture(second_moment)
            if self.fix_output_scale:
                self.log_lambda_out.zero_()
            else:
                output_energy = (
                    second_moment
                    * precision_multiplier
                    * self.log_lambda_in.exp()[None, :]
                ).sum(1)
                if self.mixture_spike_variance is None:
                    output_dimension = torch.full_like(output_energy, self.in_features)
                else:
                    output_dimension = 1.0 - self.spike_responsibility
                    output_dimension = output_dimension.sum(1)
                if self.bias_mu is not None:
                    bias_second_moment = (
                        self.bias_mu.square() + self.bias_log_variance.exp()
                    )
                    if self.mixture_spike_variance is None:
                        bias_slab_responsibility = torch.ones_like(bias_second_moment)
                    else:
                        bias_slab_responsibility = 1.0 - self.bias_spike_responsibility
                    output_energy = (
                        output_energy + bias_slab_responsibility * bias_second_moment
                    )
                    output_dimension = output_dimension + bias_slab_responsibility
                self.log_lambda_out.copy_(
                    output_dimension.clamp_min(self.variance_floor).log()
                    - output_energy.clamp_min(self.variance_floor).log()
                )
            input_energy = (
                second_moment
                * precision_multiplier
                * self.log_lambda_out.exp()[:, None]
            ).sum(0)
            if self.mixture_spike_variance is None:
                input_dimension = torch.full_like(input_energy, self.out_features)
            else:
                input_dimension = (1.0 - self.spike_responsibility).sum(0)
            self.log_lambda_in.copy_(
                input_dimension.clamp_min(self.variance_floor).log()
                - input_energy.clamp_min(self.variance_floor).log()
            )
        return self.log_lambda_out

    @torch.no_grad()
    def _update_mixture(self, second_moment: Tensor) -> Tensor:
        if self.mixture_spike_variance is None:
            return torch.ones_like(second_moment)
        base_precision = self.log_lambda_out.exp()[:, None] * self.log_lambda_in.exp()[None, :]
        probability = self.spike_probability.clamp(1e-6, 1.0 - 1e-6)
        logit_probability = probability.log() - (-probability).log1p()
        spike_precision = (-self.log_spike_variance).exp()
        responsibility_logit = (
            logit_probability
            - 0.5 * (base_precision.log() + self.log_spike_variance)
            - 0.5 * second_moment * (spike_precision - base_precision)
        )
        self.spike_responsibility.copy_(responsibility_logit.sigmoid())
        if self.bias_mu is not None:
            bias_second_moment = self.bias_mu.square() + self.bias_log_variance.exp()
            bias_base_precision = self.log_lambda_out.exp()
            bias_responsibility_logit = (
                logit_probability
                - 0.5 * (bias_base_precision.log() + self.log_spike_variance)
                - 0.5
                * bias_second_moment
                * (spike_precision - bias_base_precision)
            )
            self.bias_spike_responsibility.copy_(bias_responsibility_logit.sigmoid())
            all_responsibilities = torch.cat(
                (self.spike_responsibility.flatten(), self.bias_spike_responsibility)
            )
        else:
            all_responsibilities = self.spike_responsibility.flatten()
        self.spike_probability.copy_(
            all_responsibilities.mean().clamp(1e-6, 1.0 - 1e-6)
        )
        spike_mass = all_responsibilities.sum()
        spike_energy = (self.spike_responsibility * second_moment).sum()
        if self.bias_mu is not None:
            spike_energy = spike_energy + (
                self.bias_spike_responsibility * bias_second_moment
            ).sum()
        spike_variance = (spike_energy / spike_mass.clamp_min(self.variance_floor)).clamp_min(
            self.variance_floor
        )
        self.log_spike_variance.copy_(spike_variance.log())
        return 1.0 - self.spike_responsibility

    @torch.no_grad()
    def set_output_scale_fixed(self, fixed: bool = True) -> None:
        """Fix all output precisions at one, or restore analytical updates."""
        self.fix_output_scale = fixed
        if fixed:
            self.log_lambda_out.zero_()
        else:
            self.update_log_lambda()

    def kl_divergence(self) -> Tensor:
        base_precision = self.log_lambda_out[:, None].exp() * self.log_lambda_in[None, :].exp()
        if self.mixture_spike_variance is None:
            edge_kl = 0.5 * (
                base_precision * self.weight_second_moment()
                - 1.0
                - self.log_lambda_out[:, None]
                - self.log_lambda_in[None, :]
                - self.weight_log_variance
            )
        else:
            responsibility = self.spike_responsibility.clamp(1e-6, 1.0 - 1e-6)
            slab_responsibility = 1.0 - responsibility
            probability = self.spike_probability.clamp(1e-6, 1.0 - 1e-6)
            spike_precision = (-self.log_spike_variance).exp()
            gaussian_kl = 0.5 * (
                (responsibility * spike_precision + slab_responsibility * base_precision)
                * self.weight_second_moment()
                - 1.0
                - self.weight_log_variance
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
        total = edge_kl.sum()
        if self.bias_mu is not None:
            bias_second_moment = self.bias_mu.square() + self.bias_log_variance.exp()
            if self.mixture_spike_variance is None:
                bias_kl = 0.5 * (
                    self.log_lambda_out.exp() * bias_second_moment
                    - 1.0
                    - self.log_lambda_out
                    - self.bias_log_variance
                )
            else:
                responsibility = self.bias_spike_responsibility.clamp(1e-6, 1.0 - 1e-6)
                slab_responsibility = 1.0 - responsibility
                probability = self.spike_probability.clamp(1e-6, 1.0 - 1e-6)
                spike_precision = (-self.log_spike_variance).exp()
                bias_kl = 0.5 * (
                    (
                        responsibility * spike_precision
                        + slab_responsibility * self.log_lambda_out.exp()
                    )
                    * bias_second_moment
                    - 1.0
                    - self.bias_log_variance
                    + responsibility * self.log_spike_variance
                    - slab_responsibility * self.log_lambda_out
                )
                bias_kl = bias_kl + (
                    responsibility * (responsibility.log() - probability.log())
                    + slab_responsibility
                    * (slab_responsibility.log() - (1.0 - probability).log())
                )
            total = total + bias_kl.sum()
        return total

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
