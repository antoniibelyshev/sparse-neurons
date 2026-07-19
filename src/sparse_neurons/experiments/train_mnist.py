"""Train a row-ARD LeNet-300-100 on MNIST and plot row diagnostics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sparse_neurons.conversion import (
    ard_kl_divergence,
    convert_linear_layers,
    iter_group_ard_layers,
    update_log_lambdas,
)
from sparse_neurons.models import DeterministicLeNet300100, LeNet300100
from sparse_neurons.layers import TwoSidedGroupARDLinear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-log-variance", type=float, default=-12.0)
    parser.add_argument("--initial-relative-variance", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mnist_ard"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--kl-zero-epochs", type=int, default=5)
    parser.add_argument("--kl-warmup-epochs", type=int, default=25)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--ard-type", choices=("row", "two_sided"), default="row")
    parser.add_argument("--lr-decay-start-epoch", type=int)
    parser.add_argument("--lr-decay-gamma", type=float, default=1.0)
    parser.add_argument("--mixture-spike-ratio", type=float)
    parser.add_argument(
        "--mixture-spike-variance",
        type=float,
        help="Initial absolute spike variance; updated analytically per matrix",
    )
    parser.add_argument("--dense-final-layer", action="store_true")
    parser.add_argument("--hidden-sizes", type=int, nargs=2, default=(300, 100))
    parser.add_argument("--checkpoint-every", type=int)
    return parser.parse_args()


def kl_weight(epoch: int, zero_epochs: int, warmup_epochs: int) -> float:
    """Return a zero-then-linear KL warm-up coefficient for a 1-based epoch."""
    if epoch <= zero_epochs:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - zero_epochs) / warmup_epochs)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(
        args.data_dir, train=True, download=not args.no_download, transform=transform
    )
    test_data = datasets.MNIST(
        args.data_dir, train=False, download=not args.no_download, transform=transform
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_data, shuffle=True, **options),
        DataLoader(test_data, shuffle=False, **options),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def collect_rows(model: nn.Module, epoch: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for layer_index, layer in enumerate(iter_group_ard_layers(model), start=1):
        log_tau = -layer.log_lambda
        energy = layer.row_energy()
        for row_index in range(layer.out_features):
            rows.append(
                {
                    "epoch": epoch,
                    "layer": layer_index,
                    "row": row_index,
                    "log_lambda": layer.log_lambda[row_index].item(),
                    "log_tau": log_tau[row_index].item(),
                    "energy": energy[row_index].item(),
                }
            )
    return rows


def make_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.pretrained_checkpoint is None:
        return LeNet300100(
            initial_log_variance=args.initial_log_variance,
            hidden_sizes=tuple(args.hidden_sizes),
        ).to(device)

    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    hidden_sizes = tuple(
        checkpoint_args.get("hidden_sizes", getattr(args, "hidden_sizes", (300, 100)))
    )
    deterministic = DeterministicLeNet300100(hidden_sizes).to(device)
    deterministic.load_state_dict(checkpoint["model"])
    deterministic.eval()
    converted = convert_linear_layers(
        deterministic,
        initial_log_variance=args.initial_log_variance,
        ard_type=args.ard_type,
        initial_relative_variance=args.initial_relative_variance,
        mixture_spike_ratio=args.mixture_spike_ratio,
        mixture_spike_variance=args.mixture_spike_variance,
    )
    if args.dense_final_layer:
        converted.layers[-1] = copy.deepcopy(deterministic.layers[-1])
        print("final_layer=dense", flush=True)
    elif args.ard_type == "two_sided":
        layers = list(iter_group_ard_layers(converted))
        final_layer = layers[-1]
        assert isinstance(final_layer, TwoSidedGroupARDLinear)
        final_layer.set_output_scale_fixed()
    print(
        f"converted_pretrained={args.pretrained_checkpoint} "
        f"baseline_epoch={checkpoint.get('epoch', 'unknown')}",
        flush=True,
    )
    return converted


@torch.no_grad()
def collect_input_scales(model: nn.Module, epoch: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for layer_index, layer in enumerate(iter_group_ard_layers(model), start=1):
        if not isinstance(layer, TwoSidedGroupARDLinear):
            continue
        for input_index, value in enumerate(-layer.log_lambda_in):
            rows.append(
                {
                    "epoch": epoch,
                    "layer": layer_index,
                    "input": input_index,
                    "log_lambda_in": layer.log_lambda_in[input_index].item(),
                    "log_tau_in": value.item(),
                }
            )
    return rows


@torch.no_grad()
def collect_mixture_diagnostics(
    model: nn.Module, epoch: int
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for layer_index, layer in enumerate(iter_group_ard_layers(model), start=1):
        if not isinstance(layer, TwoSidedGroupARDLinear):
            continue
        if layer.mixture_spike_ratio is None and layer.mixture_spike_variance is None:
            continue
        responsibility = layer.spike_responsibility
        quantiles = torch.quantile(
            responsibility.flatten(), torch.tensor([0.1, 0.5, 0.9], device=responsibility.device)
        )
        rows.append(
            {
                "epoch": epoch,
                "layer": layer_index,
                "spike_probability": layer.spike_probability.item(),
                "log_spike_variance": layer.log_spike_variance.item(),
                "responsibility_p10": quantiles[0].item(),
                "responsibility_p50": quantiles[1].item(),
                "responsibility_p90": quantiles[2].item(),
                "hard_spike_fraction": (responsibility > 0.5).float().mean().item(),
            }
        )
    return rows


def write_diagnostics(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    with (output_dir / "row_diagnostics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(rows: list[dict[str, float | int]], output_dir: Path) -> None:
    epochs = sorted({int(row["epoch"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    colors = plt.cm.viridis(torch.linspace(0.15, 0.9, len(layers)).numpy())

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), squeeze=False)
    final_epoch = epochs[-1]
    for axis, layer in zip(axes[0], layers, strict=True):
        values = [float(r["log_tau"]) for r in rows if r["epoch"] == final_epoch and r["layer"] == layer]
        axis.hist(values, bins=min(40, max(10, len(values) // 5)), alpha=0.85)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel(r"$\log \tau_i = -\log \lambda_i$")
        axis.set_ylabel("Rows")
    fig.suptitle(f"Row-scale distributions after epoch {final_epoch}")
    fig.tight_layout()
    fig.savefig(output_dir / "final_log_tau_histograms.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for layer, color in zip(layers, colors, strict=True):
        medians, lower, upper = [], [], []
        for epoch in epochs:
            values = torch.tensor(
                [float(r["log_tau"]) for r in rows if r["epoch"] == epoch and r["layer"] == layer]
            )
            quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
            lower.append(quantiles[0].item())
            medians.append(quantiles[1].item())
            upper.append(quantiles[2].item())
        axis.plot(epochs, medians, color=color, label=f"Layer {layer} median")
        axis.fill_between(epochs, lower, upper, color=color, alpha=0.15)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"$\log \tau_i$")
    axis.set_title("Row-scale evolution (10th to 90th percentiles)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "log_tau_over_training.png", dpi=180)
    plt.close(fig)


def plot_accuracy_and_kl_weight(
    history: list[dict[str, float | int]],
    initial_accuracy: float,
    output_dir: Path,
) -> None:
    """Plot quality relative to the copied baseline as KL is enabled."""
    epochs = [0] + [int(row["epoch"]) for row in history]
    accuracies = [100 * initial_accuracy] + [
        100 * float(row["test_accuracy"]) for row in history
    ]
    beta = [0.0] + [float(row["kl_weight"]) for row in history]
    fig, accuracy_axis = plt.subplots(figsize=(8, 5))
    beta_axis = accuracy_axis.twinx()
    accuracy_axis.plot(epochs, accuracies, color="tab:blue", label="test accuracy")
    accuracy_axis.axhline(
        100 * initial_accuracy,
        color="tab:blue",
        linestyle="--",
        alpha=0.45,
        label="converted baseline",
    )
    beta_axis.plot(epochs, beta, color="tab:orange", label=r"KL weight $\beta$")
    accuracy_axis.set(xlabel="Epoch", ylabel="Test accuracy (%)")
    beta_axis.set(ylabel=r"KL weight $\beta$", ylim=(-0.03, 1.05))
    accuracy_axis.set_title("Accuracy during group-ARD fine-tuning")
    lines = accuracy_axis.lines + beta_axis.lines
    accuracy_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_kl_warmup.png", dpi=180)
    plt.close(fig)


def plot_two_sided_scales(
    output_rows: list[dict[str, float | int]],
    input_rows: list[dict[str, float | int]],
    output_dir: Path,
) -> None:
    """Plot the two scale vectors owned by every two-sided affine layer."""
    final_epoch = max(int(row["epoch"]) for row in output_rows)
    layers = sorted({int(row["layer"]) for row in input_rows})
    fig, axes = plt.subplots(len(layers), 1, figsize=(8, 3.5 * len(layers)), squeeze=False)
    for axis, layer in zip(axes[:, 0], layers, strict=True):
        output_values = [
            float(row["log_tau"])
            for row in output_rows
            if int(row["epoch"]) == final_epoch and int(row["layer"]) == layer
        ]
        input_values = [
            float(row["log_tau_in"])
            for row in input_rows
            if int(row["epoch"]) == final_epoch and int(row["layer"]) == layer
        ]
        axis.hist(output_values, bins=35, alpha=0.65, label="output log-tau")
        axis.hist(input_values, bins=35, alpha=0.65, label="input log-tau")
        axis.set(xlabel=r"$\log \tau$", ylabel="Count", title=f"Layer {layer}")
        axis.legend()
    fig.suptitle(f"Layer-local input/output scales after epoch {final_epoch}")
    fig.tight_layout()
    fig.savefig(output_dir / "two_sided_log_tau_histograms.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"device={device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args, device)

    model = make_model(args, device)
    initial_test_nll, initial_test_accuracy = evaluate(model, test_loader, device)
    initial_metrics = {
        "test_nll": initial_test_nll,
        "test_accuracy": initial_test_accuracy,
    }
    print(
        f"initial_test_nll={initial_test_nll:.4f} "
        f"initial_accuracy={100 * initial_test_accuracy:.2f}%",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    number_of_training_examples = len(train_loader.dataset)
    diagnostics = collect_rows(model, epoch=0)
    input_diagnostics = collect_input_scales(model, epoch=0)
    mixture_diagnostics = collect_mixture_diagnostics(model, epoch=0)
    history: list[dict[str, float | int]] = []
    if args.checkpoint_every is not None:
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "epoch": 0},
            checkpoint_dir / "epoch_000.pt",
        )

    for epoch in range(1, args.epochs + 1):
        beta = kl_weight(epoch, args.kl_zero_epochs, args.kl_warmup_epochs)
        model.train()
        train_nll_sum = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            update_log_lambdas(model)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            nll = nn.functional.cross_entropy(logits, targets)
            kl_per_example = ard_kl_divergence(model) / number_of_training_examples
            loss = nll + beta * kl_per_example
            loss.backward()
            optimizer.step()
            train_nll_sum += nll.item() * images.shape[0]

        update_log_lambdas(model)
        test_nll, test_accuracy = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_nll": train_nll_sum / number_of_training_examples,
            "test_nll": test_nll,
            "test_accuracy": test_accuracy,
            "kl_per_example": (ard_kl_divergence(model) / number_of_training_examples).item(),
            "kl_weight": beta,
        }
        history.append(record)
        diagnostics.extend(collect_rows(model, epoch))
        input_diagnostics.extend(collect_input_scales(model, epoch))
        mixture_diagnostics.extend(collect_mixture_diagnostics(model, epoch))
        print(
            f"epoch={epoch:02d} train_nll={record['train_nll']:.4f} "
            f"test_nll={test_nll:.4f} accuracy={100 * test_accuracy:.2f}% "
            f"kl/N={record['kl_per_example']:.4f} beta={beta:.3f}",
            flush=True,
        )
        if args.lr_decay_start_epoch is not None and epoch >= args.lr_decay_start_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay_gamma
        if args.checkpoint_every is not None and (
            epoch % args.checkpoint_every == 0 or epoch == args.epochs
        ):
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                args.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "initial_metrics": initial_metrics,
            "history": history,
        },
        args.output_dir / "model.pt",
    )
    with (args.output_dir / "initial_metrics.json").open("w") as file:
        json.dump(initial_metrics, file, indent=2)
    with (args.output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)
    write_diagnostics(diagnostics, args.output_dir)
    if input_diagnostics:
        with (args.output_dir / "input_scale_diagnostics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(input_diagnostics[0]))
            writer.writeheader()
            writer.writerows(input_diagnostics)
        plot_two_sided_scales(diagnostics, input_diagnostics, args.output_dir)
    if mixture_diagnostics:
        with (args.output_dir / "mixture_diagnostics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(mixture_diagnostics[0]))
            writer.writeheader()
            writer.writerows(mixture_diagnostics)
    plot_diagnostics(diagnostics, args.output_dir)
    plot_accuracy_and_kl_weight(history, initial_test_accuracy, args.output_dir)


if __name__ == "__main__":
    main()
