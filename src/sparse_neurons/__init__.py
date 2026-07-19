"""Structured variational neural-network components."""

from sparse_neurons.conversion import (
    ard_kl_divergence,
    convert_conv2d_layers,
    convert_linear_layers,
    iter_group_ard_layers,
    update_log_lambdas,
)
from sparse_neurons.layers import TwoSidedGroupARDConv2d, TwoSidedGroupARDLinear

__all__ = [
    "TwoSidedGroupARDLinear",
    "TwoSidedGroupARDConv2d",
    "ard_kl_divergence",
    "convert_linear_layers",
    "convert_conv2d_layers",
    "iter_group_ard_layers",
    "update_log_lambdas",
]
