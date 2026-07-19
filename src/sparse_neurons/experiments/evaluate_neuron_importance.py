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


@torch.no_grad()
def save_kl_diagnostics(
    layers: list[TwoSidedGroupARDLinear], output_dir: Path
) -> None:
    """Visualize and tabulate the exact augmented-weight KL decomposition."""
    fig, axes = plt.subplots(
        len(layers), 3, figsize=(18, 5 * len(layers)), squeeze=False
    )
    top_rows: list[dict[str, float | int]] = []
    for layer_index, (axis_row, layer) in enumerate(
        zip(axes, layers, strict=True), 1
    ):
        kl = layer.elementwise_kl_divergence().detach().cpu()
        log_kl = kl.clamp_min(1e-12).log10()

        row_order = kl.sum(1).argsort(descending=True)
        column_order = kl.sum(0).argsort(descending=True)
        sorted_log_kl = log_kl[row_order][:, column_order]
        image = axis_row[0].imshow(sorted_log_kl.numpy(), aspect="auto", cmap="magma")
        axis_row[0].set(
            xlabel="Input rank by total KL",
            ylabel="Neuron rank by total KL",
            title=rf"Layer {layer_index}: sorted $\log_{{10}} K_{{ji}}$",
        )
        fig.colorbar(image, ax=axis_row[0], label=r"$\log_{10} K_{ji}$")

        axis_row[1].hist(log_kl.flatten().numpy(), bins=80, alpha=0.85)
        axis_row[1].set(
            xlabel=r"$\log_{10} K_{ji}$",
            ylabel="Augmented weights",
            title=f"Layer {layer_index}: KL contribution distribution",
        )
        axis_row[1].grid(axis="y", alpha=0.2)

        flat_kl = kl.flatten()
        descending_kl, flat_order = flat_kl.sort(descending=True)
        fraction_weights = torch.arange(1, flat_kl.numel() + 1) / flat_kl.numel()
        cumulative_kl = descending_kl.cumsum(0) / descending_kl.sum().clamp_min(1e-30)
        axis_row[2].plot(fraction_weights.numpy(), cumulative_kl.numpy())
        axis_row[2].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
        for target, color in ((0.5, "tab:orange"), (0.9, "tab:red")):
            count = int(torch.searchsorted(cumulative_kl, target).item()) + 1
            fraction = count / flat_kl.numel()
            axis_row[2].axvline(
                fraction,
                color=color,
                linestyle=":",
                label=f"{100 * target:.0f}% KL from {100 * fraction:.1f}% weights",
            )
        axis_row[2].set(
            xlabel="Fraction of augmented weights (largest first)",
            ylabel="Fraction of total KL",
            title=f"Layer {layer_index}: KL concentration",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis_row[2].grid(alpha=0.2)
        axis_row[2].legend()

        means = layer.augmented_weight_mu().detach().cpu()
        stds = layer.augmented_weight_log_std().exp().detach().cpu()
        responsibilities = layer.spike_responsibility.detach().cpu()
        slab_variance = (
            -layer.log_lambda_out[:, None] - layer.log_lambda_in[None, :]
        ).exp().detach().cpu()
        for rank, flat_index in enumerate(flat_order[: min(200, flat_order.numel())], 1):
            output_index = int(flat_index // layer.augmented_in_features)
            input_index = int(flat_index % layer.augmented_in_features)
            top_rows.append(
                {
                    "layer": layer_index,
                    "rank": rank,
                    "output": output_index,
                    "input": input_index,
                    "is_bias": int(
                        layer.bias_mu is not None and input_index == layer.in_features
                    ),
                    "kl": kl[output_index, input_index].item(),
                    "mu": means[output_index, input_index].item(),
                    "std": stds[output_index, input_index].item(),
                    "spike_responsibility": responsibilities[
                        output_index, input_index
                    ].item(),
                    "slab_variance": slab_variance[output_index, input_index].item(),
                    "spike_variance": layer.log_spike_variance.exp().item(),
                }
            )

    fig.suptitle("Per-weight variational KL diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_kl_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(layers), 1, figsize=(14, 4.5 * len(layers)), squeeze=False
    )
    for layer_index, axis in enumerate(axes[:, 0], 1):
        rows = [row for row in top_rows if row["layer"] == layer_index][:40]
        labels = [
            f"{row['output']}:" + ("bias" if row["is_bias"] else str(row["input"]))
            for row in rows
        ]
        colors = ["tab:red" if row["is_bias"] else "tab:blue" for row in rows]
        axis.bar(range(1, len(rows) + 1), [row["kl"] for row in rows], color=colors)
        axis.set_xticks(range(1, len(rows) + 1), labels, rotation=90)
        axis.set(
            xlabel="Augmented weight (output:input)",
            ylabel=r"$K_{ji}$",
            title=f"Layer {layer_index}: top 40 per-weight KL contributors",
        )
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "top_weight_kl_contributors.png", dpi=180)
    plt.close(fig)

    with (output_dir / "top_weight_kl_contributors.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(top_rows[0]))
        writer.writeheader()
        writer.writerows(top_rows)


def save_importance(model: nn.Module, output_dir: Path) -> None:
    layers = [
        layer
        for layer in iter_group_ard_layers(model)
        if isinstance(layer, TwoSidedGroupARDLinear)
    ]
    if not layers:
        raise ValueError("Checkpoint contains no two-sided group-ARD layers")
    save_weight_log_snr_histograms(layers, output_dir)
    save_kl_diagnostics(layers, output_dir)

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
