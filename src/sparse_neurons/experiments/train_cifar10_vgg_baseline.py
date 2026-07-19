"""Train an EMA VGG-11 baseline on CIFAR-10."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sparse_neurons.ema import ParameterEMA
from sparse_neurons.experiments.train_mnist import choose_device
from sparse_neurons.models import DeterministicVGG11Cifar10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cifar10_vgg/baseline"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def make_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_data = datasets.CIFAR10(
        args.data_dir, train=True, download=not args.no_download, transform=train_transform
    )
    test_data = datasets.CIFAR10(
        args.data_dir, train=False, download=not args.no_download, transform=test_transform
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"device={device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args, device)
    model = DeterministicVGG11Cifar10().to(device)
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
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), targets)
            loss.backward()
            optimizer.step()
            ema.update(model)
            train_loss += loss.item() * images.shape[0]
        scheduler.step()
        ema.copy_to(ema_model)
        test_nll, test_accuracy = evaluate(ema_model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_nll": train_loss / len(train_loader.dataset),
            "test_nll": test_nll,
            "test_accuracy": test_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_nll={record['train_nll']:.4f} "
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
            "architecture": "vgg11_cifar10",
        },
        args.output_dir / "final_ema_model.pt",
    )
    with (args.output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)


if __name__ == "__main__":
    main()
