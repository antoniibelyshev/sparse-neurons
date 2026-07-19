"""Utilities for converting deterministic Torch networks to group ARD."""

from __future__ import annotations

import copy
from collections.abc import Iterator

from torch import nn

from sparse_neurons.layers import TwoSidedGroupARDConv2d, TwoSidedGroupARDLinear

ARDLayer = TwoSidedGroupARDLinear | TwoSidedGroupARDConv2d


def convert_linear_layers(
    module: nn.Module,
    *,
    copy_module: bool = True,
    variance_floor: float = 1e-12,
    initial_relative_std: float = 1e-2,
    mixture_spike_variance: float = 1e-4,
) -> nn.Module:
    """Replace every ``nn.Linear`` with the selected augmented ARD layer.

    By default the complete network is deep-copied, so the input model remains
    unchanged. All non-linear modules and their state are preserved. Linear
    weights and biases are copied into posterior means, preserving the network's
    deterministic output when the converted model is evaluated with posterior
    means. Shared references to the same linear module remain shared.

    Args:
        module: A standard Torch module, potentially with nested containers.
        copy_module: Deep-copy ``module`` before replacing layers.
        variance_floor: Numerical variance floor.
    """
    result = copy.deepcopy(module) if copy_module else module
    replacements: dict[int, TwoSidedGroupARDLinear] = {}

    def convert_children(parent: nn.Module) -> None:
        for name, child in tuple(parent._modules.items()):
            if child is None:
                continue
            if isinstance(child, TwoSidedGroupARDLinear):
                continue
            if isinstance(child, nn.Linear):
                key = id(child)
                replacement = replacements.get(key)
                if replacement is None:
                    common = {
                        "variance_floor": variance_floor,
                        "initial_relative_std": initial_relative_std,
                    }
                    replacement = TwoSidedGroupARDLinear.from_linear(
                        child,
                        mixture_spike_variance=mixture_spike_variance,
                        **common,
                    )
                    replacements[key] = replacement
                parent._modules[name] = replacement
            else:
                convert_children(child)

    if isinstance(result, nn.Linear):
        common = {
            "variance_floor": variance_floor,
            "initial_relative_std": initial_relative_std,
        }
        return TwoSidedGroupARDLinear.from_linear(
            result,
            mixture_spike_variance=mixture_spike_variance,
            **common,
        )
    convert_children(result)
    return result


def convert_conv2d_layers(
    module: nn.Module,
    *,
    copy_module: bool = True,
    variance_floor: float = 1e-12,
    initial_relative_std: float = 1e-2,
    mixture_spike_variance: float = 1e-4,
) -> nn.Module:
    """Replace standard convolutions with channel-structured ARD convolutions."""
    result = copy.deepcopy(module) if copy_module else module
    replacements: dict[int, TwoSidedGroupARDConv2d] = {}

    def convert_children(parent: nn.Module) -> None:
        for name, child in tuple(parent._modules.items()):
            if child is None or isinstance(child, TwoSidedGroupARDConv2d):
                continue
            if isinstance(child, nn.Conv2d):
                key = id(child)
                replacement = replacements.get(key)
                if replacement is None:
                    replacement = TwoSidedGroupARDConv2d.from_conv2d(
                        child,
                        variance_floor=variance_floor,
                        initial_relative_std=initial_relative_std,
                        mixture_spike_variance=mixture_spike_variance,
                    )
                    replacements[key] = replacement
                parent._modules[name] = replacement
            else:
                convert_children(child)

    if isinstance(result, nn.Conv2d):
        return TwoSidedGroupARDConv2d.from_conv2d(
            result,
            variance_floor=variance_floor,
            initial_relative_std=initial_relative_std,
            mixture_spike_variance=mixture_spike_variance,
        )
    convert_children(result)
    return result


def iter_group_ard_layers(module: nn.Module) -> Iterator[ARDLayer]:
    """Iterate over all group-ARD layers in a module tree."""
    for child in module.modules():
        if isinstance(child, (TwoSidedGroupARDLinear, TwoSidedGroupARDConv2d)):
            yield child


def update_log_lambdas(module: nn.Module) -> None:
    """Apply the analytical M-step to every group-ARD layer."""
    for layer in iter_group_ard_layers(module):
        layer.update_log_lambda()


def ard_kl_divergence(module: nn.Module):
    """Return the total KL divergence of every group-ARD layer."""
    return sum(layer.kl_divergence() for layer in iter_group_ard_layers(module))
