"""Plot maximum augmented-weight SNR neuron importance."""

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


def neuron_importance(layer: TwoSidedGroupARDLinear) -> torch.Tensor:
    """Return maximum posterior SNR, treating bias as a constant-input weight."""
    variance = (2.0 * layer.augmented_weight_log_std()).exp()
    weight_snr = layer.augmented_weight_mu().square() / variance.clamp_min(
        layer.variance_floor
    )
    return weight_snr.amax(1)


@torch.no_grad()
def save_weight_log_snr_histograms(
    layers: list[TwoSidedGroupARDLinear], output_dir: Path
) -> None:
    """Plot the distribution of log posterior SNR for every augmented weight."""
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4), squeeze=False)
    for layer_index, (axis, layer) in enumerate(zip(axes[0], layers, strict=True), 1):
        means = layer.augmented_weight_mu().flatten()
        log_variances = 2.0 * layer.augmented_weight_log_std().flatten()
        squared_mean = means.square()
        log_snr = (
            squared_mean.clamp_min(torch.finfo(squared_mean.dtype).tiny).log()
            - log_variances
        ).cpu()
        axis.hist(log_snr.numpy(), bins=80, alpha=0.85)
        axis.set(
            xlabel=r"$\log(\mu_{ji}^2/s_{ji}^2)$",
            ylabel="Weights",
            title=f"Hidden layer {layer_index} ({log_snr.numel():,} weights)",
        )
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Weight-level posterior log-SNR distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_log_snr_histograms.png", dpi=180)
    plt.close(fig)


def save_importance(model: nn.Module, output_dir: Path) -> None:
    layers = [
        layer
        for layer in iter_group_ard_layers(model)
        if isinstance(layer, TwoSidedGroupARDLinear)
    ]
    if not layers:
        raise ValueError("Checkpoint contains no two-sided group-ARD layers")
    save_weight_log_snr_histograms(layers, output_dir)

    rows: list[dict[str, float | int]] = []
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 4 * len(layers)), squeeze=False)
    for layer_index, (axis, layer) in enumerate(zip(axes[:, 0], layers, strict=True), 1):
        importance = neuron_importance(layer)
        importance = importance.detach().cpu()
        order = importance.argsort(descending=True)
        for rank, neuron in enumerate(order.tolist(), 1):
            rows.append(
                {
                    "layer": layer_index,
                    "rank": rank,
                    "neuron": neuron,
                    "importance": importance[neuron].item(),
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
            ylabel=r"Maximum augmented-weight SNR $\max_i \mu_{ji}^2/s_{ji}^2$",
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
