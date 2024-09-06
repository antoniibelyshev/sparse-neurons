import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST, CIFAR10

from typing import Any
import os


data_dir = os.path.join(os.getcwd(), "data")


class FlattenTransform:
    def __call__(self, x: Tensor) -> Tensor:
        return x.view(-1)


class RGB2YUVTransform:
    @staticmethod
    def __call__(rgb: Tensor) -> Tensor:     
        m = torch.tensor([[0.29900, -0.16874,  0.50000],
                          [0.58700, -0.33126, -0.41869],
                          [0.11400,  0.50000, -0.08131]])
        
        yuv = (rgb.permute(1, 2, 0) @ m).permute(2, 0, 1)
        yuv[1:, :, :] += 0.5
        return yuv


class Normalization:
    m: Tensor
    s: Tensor

    def __init__(self, kernel_size: int = 7, sigma: float = 1.0) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma

    @property
    def gaussian_kernel(self) -> Tensor:
        x = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / self.sigma).pow(2))
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d /= gaussian_2d.sum()
        return gaussian_2d.expand(1, 1, -1, -1)

    def fit(self, imgs: Tensor) -> None:
        uv_channels = imgs[:, 1:]
        self.m = uv_channels.mean(dim=(0, 2, 3), keepdims=True) # type: ignore
        self.s = uv_channels.std(dim=(0, 2, 3), keepdims=True) # type: ignore

    def transform(self, imgs: Tensor) -> Tensor:
        y_channel = imgs[:, :1]
        uv_channels = imgs[:, 1:]
        
        y_blurred = torch.nn.functional.conv2d(y_channel, self.gaussian_kernel, padding=self.kernel_size // 2)
        y_normalized = y_channel - y_blurred
        y_normalized /= y_normalized.std((2, 3), keepdims=True) + 1e-5 # type: ignore
        
        uv_normalized = (uv_channels - self.m) / self.s

        return torch.cat([y_normalized, uv_normalized], 1)

    def fit_transform(self, imgs: Tensor) -> Tensor:
        self.fit(imgs)
        return self.transform(imgs)


def get_images(dataset: Dataset[tuple[Tensor, Tensor]]) -> Tensor:
    return torch.stack([el[0] for el in dataset])


def get_targets(dataset: Dataset[tuple[Tensor, Tensor]]) -> Tensor:
    return torch.tensor([el[1] for el in dataset])


def get_mnist_dataloaders(
        batch_size_train: int = 64,
        batch_size_test: int = 1000,
        **train_kwargs: Any
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    transform = Compose([ # type: ignore
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        FlattenTransform(),
    ])
    train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, **train_kwargs)
    test_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader # type: ignore


def get_cifar_dataloaders(
        batch_size_train: int = 128,
        batch_size_test: int = 1000,
        **train_kwargs: Any
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    transform = Compose([
        ToTensor(),
        RGB2YUVTransform(),
    ])


    raw_train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    raw_test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    norm = Normalization()
    train_dataset = TensorDataset(norm.fit_transform(get_images(raw_train_dataset)), get_targets(raw_train_dataset))
    test_dataset = TensorDataset(norm.transform(get_images(raw_test_dataset)), get_targets(raw_test_dataset))

    train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, **train_kwargs)
    test_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader # type: ignore


def get_dataloaders(name: str, **kwargs: Any) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    if name == "MNIST":
        return get_mnist_dataloaders(**kwargs)
    elif name == "CIFAR":
        return get_cifar_dataloaders(**kwargs)
    else:
        raise AssertionError(f"Unsupported dataset: {name}")
