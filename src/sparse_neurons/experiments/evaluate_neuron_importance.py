"""Plot the selected model's slab-conditioned neuron importance."""

from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from sparse_neurons.conversion import iter_group_ard_layers
from sparse_neurons.experiments.train_mnist import choose_device, make_model
from sparse_neurons.layers import TwoSidedGroupARDLinear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/neuron_importance"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = Namespace(**checkpoint["args"])
    model = make_model(args, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def neuron_importance(layer: TwoSidedGroupARDLinear) -> tuple[torch.Tensor, torch.Tensor]:
    """Return slab-conditioned row SNR and effective slab support."""
    slab_probability = 1.0 - layer.spike_responsibility
    signal = (slab_probability * layer.weight_mu.square()).sum(1)
    noise = (slab_probability * layer.weight_log_variance.exp()).sum(1)
    if layer.bias_mu is not None:
        signal = signal + layer.bias_mu.square()
        noise = noise + layer.bias_log_variance.exp()
    importance = signal / noise.clamp_min(layer.variance_floor)
    return importance, slab_probability.sum(1)


def save_importance(model: nn.Module, output_dir: Path) -> None:
    layers = [
        layer
        for layer in iter_group_ard_layers(model)
        if isinstance(layer, TwoSidedGroupARDLinear)
    ]
    if not layers:
        raise ValueError("Checkpoint contains no two-sided group-ARD layers")

    rows: list[dict[str, float | int]] = []
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 4 * len(layers)), squeeze=False)
    for layer_index, (axis, layer) in enumerate(zip(axes[:, 0], layers, strict=True), 1):
        importance, support = neuron_importance(layer)
        importance = importance.detach().cpu()
        support = support.detach().cpu()
        order = importance.argsort(descending=True)
        for rank, neuron in enumerate(order.tolist(), 1):
            rows.append(
                {
                    "layer": layer_index,
                    "rank": rank,
                    "neuron": neuron,
                    "importance": importance[neuron].item(),
                    "slab_support": support[neuron].item(),
                }
            )
        axis.bar(
            torch.arange(1, len(order) + 1).numpy(),
            importance[order].clamp_min(1e-30).numpy(),
            width=1.0,
        )
        axis.set_yscale("log")
        axis.set(
            xlabel="Importance rank (most to least important)",
            ylabel="Slab-conditioned row SNR",
            title=f"Hidden layer {layer_index}",
        )
        axis.grid(axis="y", which="both", alpha=0.2)
    fig.suptitle("Selected model: sorted neuron importance")
    fig.tight_layout()
    fig.savefig(output_dir / "sorted_neuron_importance.png", dpi=180)
    plt.close(fig)

    with (output_dir / "neuron_importance.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, choose_device(args.device))
    save_importance(model, output_dir)


if __name__ == "__main__":
    main()
