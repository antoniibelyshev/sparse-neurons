"""Utilities for converting deterministic Torch networks to group ARD."""

from __future__ import annotations

import copy
from collections.abc import Iterator

from torch import nn

from sparse_neurons.layers import GroupARDLinear, TwoSidedGroupARDLinear

ARDLinear = GroupARDLinear | TwoSidedGroupARDLinear


def convert_linear_layers(
    module: nn.Module,
    *,
    copy_module: bool = True,
    initial_log_variance: float = -12.0,
    a0: float | None = None,
    b0: float | None = None,
    variance_floor: float = 1e-12,
    ard_type: str = "row",
    initial_relative_variance: float | None = None,
    mixture_spike_variance: float | None = None,
) -> nn.Module:
    """Replace every ``nn.Linear`` in a network with ``GroupARDLinear``.

    By default the complete network is deep-copied, so the input model remains
    unchanged. All non-linear modules and their state are preserved. Linear
    weights and biases are copied into posterior means, preserving the network's
    deterministic output when the converted model is evaluated with posterior
    means. Shared references to the same linear module remain shared.

    Args:
        module: A standard Torch module, potentially with nested containers.
        copy_module: Deep-copy ``module`` before replacing layers.
        initial_log_variance: Initial posterior log variance in converted layers.
        a0: Optional Gamma-prior shape for all converted layers.
        b0: Optional Gamma-prior rate for all converted layers.
        variance_floor: Numerical variance floor.
    """
    result = copy.deepcopy(module) if copy_module else module
    if ard_type not in {"row", "two_sided"}:
        raise ValueError("ard_type must be 'row' or 'two_sided'")
    layer_class = {
        "row": GroupARDLinear,
        "two_sided": TwoSidedGroupARDLinear,
    }[ard_type]
    replacements: dict[int, ARDLinear] = {}

    def convert_children(parent: nn.Module) -> None:
        for name, child in tuple(parent._modules.items()):
            if child is None:
                continue
            if isinstance(child, (GroupARDLinear, TwoSidedGroupARDLinear)):
                continue
            if isinstance(child, nn.Linear):
                key = id(child)
                replacement = replacements.get(key)
                if replacement is None:
                    common = {
                        "initial_log_variance": initial_log_variance,
                        "variance_floor": variance_floor,
                        "initial_relative_variance": initial_relative_variance,
                    }
                    if layer_class is GroupARDLinear:
                        replacement = layer_class.from_linear(child, a0=a0, b0=b0, **common)
                    elif layer_class is TwoSidedGroupARDLinear:
                        replacement = layer_class.from_linear(
                            child,
                            mixture_spike_variance=mixture_spike_variance,
                            **common,
                        )
                    replacements[key] = replacement
                parent._modules[name] = replacement
            else:
                convert_children(child)

    if isinstance(result, nn.Linear) and not isinstance(result, ARDLinear):
        common = {
            "initial_log_variance": initial_log_variance,
            "variance_floor": variance_floor,
            "initial_relative_variance": initial_relative_variance,
        }
        if layer_class is GroupARDLinear:
            return layer_class.from_linear(result, a0=a0, b0=b0, **common)
        return layer_class.from_linear(
            result,
            mixture_spike_variance=mixture_spike_variance,
            **common,
        )
    convert_children(result)
    return result


def iter_group_ard_layers(module: nn.Module) -> Iterator[ARDLinear]:
    """Iterate over all group-ARD layers in a module tree."""
    for child in module.modules():
        if isinstance(child, (GroupARDLinear, TwoSidedGroupARDLinear)):
            yield child


def update_log_lambdas(module: nn.Module) -> None:
    """Apply the analytical M-step to every group-ARD layer."""
    for layer in iter_group_ard_layers(module):
        layer.update_log_lambda()


def ard_kl_divergence(module: nn.Module):
    """Return the total KL divergence of every group-ARD layer."""
    return sum(layer.kl_divergence() for layer in iter_group_ard_layers(module))
