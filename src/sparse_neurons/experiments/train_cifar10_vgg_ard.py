"""Fine-tune an EMA VGG-11 with channel-structured ARD on CIFAR-10."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch import nn

from sparse_neurons.conversion import (
    ard_kl_divergence,
    convert_conv2d_layers,
    update_log_lambdas,
)
from sparse_neurons.ema import ParameterEMA
from sparse_neurons.experiments.train_cifar10_vgg_baseline import evaluate, make_loaders
from sparse_neurons.experiments.train_mnist import choose_device, kl_weight
from sparse_neurons.models import DeterministicVGG11Cifar10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--initial-relative-std", type=float, default=1e-2)
    parser.add_argument("--mixture-spike-variance", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cifar10_vgg/ard"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def make_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    checkpoint = torch.load(
        args.pretrained_checkpoint, map_location=device, weights_only=False
    )
    deterministic = DeterministicVGG11Cifar10().to(device)
    deterministic.load_state_dict(checkpoint["model"])
    deterministic.eval()
    converted = convert_conv2d_layers(
        deterministic,
        initial_relative_std=args.initial_relative_std,
        mixture_spike_variance=args.mixture_spike_variance,
    )
    converted.classifier = copy.deepcopy(deterministic.classifier)
    print(
        f"converted_pretrained={args.pretrained_checkpoint} "
        f"baseline_epoch={checkpoint.get('epoch', 'unknown')} classifier=dense",
        flush=True,
    )
    return converted


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"device={device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args, device)
    model = make_model(args, device)
    initial_nll, initial_accuracy = evaluate(model, test_loader, device)
    print(
        f"initial_test_nll={initial_nll:.4f} "
        f"initial_accuracy={100 * initial_accuracy:.2f}%",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 100
    )
    ema = ParameterEMA(model, args.ema_decay)
    ema_model = copy.deepcopy(model)
    number_of_examples = len(train_loader.dataset)
    history: list[dict[str, float | int]] = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        beta = kl_weight(epoch, 0, args.kl_warmup_epochs)
        model.train()
        train_nll = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            update_log_lambdas(model)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            nll = nn.functional.cross_entropy(logits, targets)
            kl_per_example = ard_kl_divergence(model) / number_of_examples
            loss = nll + beta * kl_per_example
            loss.backward()
            optimizer.step()
            ema.update(model)
            train_nll += nll.item() * images.shape[0]
        scheduler.step()
        update_log_lambdas(model)
        ema.copy_to(ema_model)
        update_log_lambdas(ema_model)
        test_nll, test_accuracy = evaluate(ema_model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_nll": train_nll / number_of_examples,
            "test_nll": test_nll,
            "test_accuracy": test_accuracy,
            "kl_per_example": (
                ard_kl_divergence(ema_model) / number_of_examples
            ).item(),
            "kl_weight": beta,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_nll={record['train_nll']:.4f} "
            f"test_nll={test_nll:.4f} accuracy={100 * test_accuracy:.2f}% "
            f"kl/N={record['kl_per_example']:.4f} beta={beta:.3f} "
            f"lr={record['learning_rate']:.6f}",
            flush=True,
        )
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save(
                {"model": ema_model.state_dict(), "args": vars(args), "epoch": epoch},
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )

    torch.save(
        {
            "model": ema_model.state_dict(),
            "args": vars(args),
            "history": history,
            "epoch": args.epochs,
            "architecture": "vgg11_cifar10_ard",
        },
        args.output_dir / "model.pt",
    )
    with (args.output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)


if __name__ == "__main__":
    main()
