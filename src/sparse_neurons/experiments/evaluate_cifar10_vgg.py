"""Generate channel-importance and per-weight KL diagnostics for ARD VGG-11."""

from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sparse_neurons.conversion import iter_group_ard_layers
from sparse_neurons.experiments.train_cifar10_vgg_ard import make_model
from sparse_neurons.experiments.train_mnist import choose_device
from sparse_neurons.layers import TwoSidedGroupARDConv2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cifar10_vgg/diagnostics"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = Namespace(**checkpoint["args"])
    model = make_model(model_args, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    layers = [
        layer
        for layer in iter_group_ard_layers(model)
        if isinstance(layer, TwoSidedGroupARDConv2d)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.output_dir / "kl_layers"
    heatmap_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), squeeze=False)
    channel_rows: list[dict[str, float | int]] = []
    top_weight_rows: list[dict[str, float | int]] = []
    for layer_index, (axis, layer) in enumerate(zip(axes.flat, layers, strict=True), 1):
        means = layer.augmented_weight_mu()
        stds = layer.augmented_weight_log_std().exp()
        snr = means.square() / stds.square().clamp_min(layer.variance_floor)
        importance = snr.amax(1).cpu()
        order = importance.argsort(descending=True)
        axis.bar(range(1, layer.out_channels + 1), importance[order].numpy(), width=1.0)
        axis.set_yscale("log")
        axis.set(
            xlabel="Output-channel importance rank",
            ylabel=r"$\max_i \mu_{ji}^2/s_{ji}^2$",
            title=f"Conv layer {layer_index}: {layer.in_channels} to {layer.out_channels}",
        )
        axis.grid(axis="y", which="both", alpha=0.2)

        elementwise_kl = layer.elementwise_kl_divergence().cpu()
        channel_kl = elementwise_kl.sum(1)
        for rank, output_channel in enumerate(order.tolist(), 1):
            channel_rows.append(
                {
                    "layer": layer_index,
                    "rank": rank,
                    "output_channel": output_channel,
                    "importance": importance[output_channel].item(),
                    "total_kl": channel_kl[output_channel].item(),
                    "mean_spike_responsibility": layer.spike_responsibility[
                        output_channel
                    ].mean().item(),
                }
            )

        flat_kl = elementwise_kl.flatten()
        descending_kl, flat_order = flat_kl.sort(descending=True)
        cumulative = descending_kl.cumsum(0) / descending_kl.sum().clamp_min(1e-30)
        fraction = torch.arange(1, flat_kl.numel() + 1) / flat_kl.numel()
        row_order = channel_kl.argsort(descending=True)
        column_order = elementwise_kl.sum(0).argsort(descending=True)
        log_kl = elementwise_kl.clamp_min(1e-12).log10()

        detail_fig, detail_axes = plt.subplots(1, 3, figsize=(18, 5))
        image = detail_axes[0].imshow(
            log_kl[row_order][:, column_order].numpy(), aspect="auto", cmap="magma"
        )
        detail_fig.colorbar(image, ax=detail_axes[0], label=r"$\log_{10}K_{ji}$")
        detail_axes[0].set(title="Sorted KL heatmap", xlabel="Scalar input rank", ylabel="Channel rank")
        detail_axes[1].hist(log_kl.flatten().numpy(), bins=80)
        detail_axes[1].set(title="Weight KL distribution", xlabel=r"$\log_{10}K_{ji}$", ylabel="Weights")
        detail_axes[2].plot(fraction.numpy(), cumulative.numpy())
        detail_axes[2].plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
        for target, color in ((0.5, "tab:orange"), (0.9, "tab:red")):
            count = int(torch.searchsorted(cumulative, target).item()) + 1
            x = count / flat_kl.numel()
            detail_axes[2].axvline(x, color=color, linestyle=":", label=f"{100*target:.0f}% from {100*x:.1f}%")
        detail_axes[2].set(title="KL concentration", xlabel="Largest-weight fraction", ylabel="Total-KL fraction", xlim=(0, 1), ylim=(0, 1))
        detail_axes[2].legend()
        detail_fig.suptitle(f"Convolution layer {layer_index} KL diagnostics")
        detail_fig.tight_layout()
        detail_fig.savefig(heatmap_dir / f"layer_{layer_index:02d}.png", dpi=160)
        plt.close(detail_fig)

        for rank, flat_index in enumerate(flat_order[:100], 1):
            output_channel = int(flat_index // layer.augmented_columns)
            column = int(flat_index % layer.augmented_columns)
            is_bias = int(layer.bias_mu is not None and column == layer.augmented_columns - 1)
            input_channel = -1 if is_bias else column // layer.kernel_elements
            kernel_offset = -1 if is_bias else column % layer.kernel_elements
            top_weight_rows.append(
                {
                    "layer": layer_index,
                    "rank": rank,
                    "output_channel": output_channel,
                    "input_channel": input_channel,
                    "kernel_row": -1 if is_bias else kernel_offset // layer.kernel_size[1],
                    "kernel_column": -1 if is_bias else kernel_offset % layer.kernel_size[1],
                    "is_bias": is_bias,
                    "kl": elementwise_kl[output_channel, column].item(),
                    "mu": means[output_channel, column].item(),
                    "std": stds[output_channel, column].item(),
                    "spike_responsibility": layer.spike_responsibility[
                        output_channel, column
                    ].item(),
                }
            )

    fig.suptitle("VGG-11 sorted output-channel importance")
    fig.tight_layout()
    fig.savefig(args.output_dir / "sorted_channel_importance.png", dpi=180)
    plt.close(fig)

    for filename, rows in (
        ("channel_importance.csv", channel_rows),
        ("top_weight_kl_contributors.csv", top_weight_rows),
    ):
        with (args.output_dir / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
