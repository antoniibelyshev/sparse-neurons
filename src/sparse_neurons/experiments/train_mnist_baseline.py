"""Train a deterministic LeNet-300-100 baseline on MNIST."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from sparse_neurons.experiments.train_mnist import choose_device, make_loaders
from sparse_neurons.ema import ParameterEMA
from sparse_neurons.models import DeterministicLeNet300100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mnist_baseline"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--hidden-sizes", type=int, nargs=2, default=(300, 100))
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def plot_history(history: list[dict[str, float | int]], output_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_nll"]) for row in history], label="train")
    axes[0].plot(epochs, [float(row["test_nll"]) for row in history], label="test")
    axes[0].set(xlabel="Epoch", ylabel="NLL", title="Negative log-likelihood")
    axes[0].legend()
    axes[1].plot(epochs, [100 * float(row["test_accuracy"]) for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Test accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"device={device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args, device)
    model = DeterministicLeNet300100(tuple(args.hidden_sizes)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 100
    )
    ema = ParameterEMA(model, args.ema_decay)
    ema_model = copy.deepcopy(model)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), targets)
            loss.backward()
            optimizer.step()
            ema.update(model)
            train_loss_sum += loss.item() * images.shape[0]
        scheduler.step()

        ema.copy_to(ema_model)
        test_nll, test_accuracy = evaluate(ema_model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_nll": train_loss_sum / len(train_loader.dataset),
            "test_nll": test_nll,
            "test_accuracy": test_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} train_nll={record['train_nll']:.4f} "
            f"test_nll={test_nll:.4f} accuracy={100 * test_accuracy:.2f}% "
            f"lr={record['learning_rate']:.6f}",
            flush=True,
        )

    torch.save(
        {
            "model": ema_model.state_dict(),
            "args": vars(args),
            "history": history,
            "epoch": args.epochs,
        },
        args.output_dir / "final_ema_model.pt",
    )
    with (args.output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)
    plot_history(history, args.output_dir)


if __name__ == "__main__":
    main()
