"""Structured variational neural-network components."""

from sparse_neurons.conversion import (
    ard_kl_divergence,
    convert_linear_layers,
    iter_group_ard_layers,
    update_log_lambdas,
)
from sparse_neurons.layers import GroupARDLinear, TwoSidedGroupARDLinear

__all__ = [
    "GroupARDLinear",
    "TwoSidedGroupARDLinear",
    "ard_kl_divergence",
    "convert_linear_layers",
    "iter_group_ard_layers",
    "update_log_lambdas",
]
