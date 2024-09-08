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
        """
        Flattens the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Flattened tensor.
        """
        return x.view(-1)


class RGB2YUVTransform:
    @staticmethod
    def __call__(rgb: Tensor) -> Tensor:
        """
        Converts RGB image tensor to YUV color space.

        Args:
            rgb (Tensor): RGB image tensor.

        Returns:
            Tensor: YUV image tensor.
        """
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
        """
        Initializes Normalization with Gaussian kernel parameters.

        Args:
            kernel_size (int): Size of the Gaussian kernel. Default is 7.
            sigma (float): Standard deviation of the Gaussian kernel. Default is 1.0.
        """
        self.kernel_size = kernel_size
        self.sigma = sigma

    @property
    def gaussian_kernel(self) -> Tensor:
        """
        Computes the Gaussian kernel.

        Returns:
            Tensor: 2D Gaussian kernel.
        """
        x = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / self.sigma).pow(2))
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d /= gaussian_2d.sum()
        return gaussian_2d.expand(1, 1, -1, -1)

    def fit(self, imgs: Tensor) -> None:
        """
        Computes normalization parameters from the input images.

        Args:
            imgs (Tensor): Input image tensor.
        """
        uv_channels = imgs[:, 1:]
        self.m = uv_channels.mean(dim=(0, 2, 3), keepdims=True)
        self.s = uv_channels.std(dim=(0, 2, 3), keepdims=True)

    def transform(self, imgs: Tensor) -> Tensor:
        """
        Applies normalization to the input images.

        Args:
            imgs (Tensor): Input image tensor.

        Returns:
            Tensor: Normalized image tensor.
        """
        y_channel = imgs[:, :1]
        uv_channels = imgs[:, 1:]
        
        y_blurred = torch.nn.functional.conv2d(y_channel, self.gaussian_kernel, padding=self.kernel_size // 2)
        y_normalized = y_channel - y_blurred
        y_normalized /= y_normalized.std((2, 3), keepdims=True) + 1e-5
        
        uv_normalized = (uv_channels - self.m) / self.s

        return torch.cat([y_normalized, uv_normalized], 1)

    def fit_transform(self, imgs: Tensor) -> Tensor:
        """
        Fits normalization parameters and applies normalization.

        Args:
            imgs (Tensor): Input image tensor.

        Returns:
            Tensor: Normalized image tensor.
        """
        self.fit(imgs)
        return self.transform(imgs)


def get_images(dataset: Dataset[tuple[Tensor, Tensor]]) -> Tensor:
    """
    Extracts images from a dataset.

    Args:
        dataset (Dataset): Dataset of images and labels.

    Returns:
        Tensor: Tensor of images.
    """
    return torch.stack([el[0] for el in dataset])


def get_targets(dataset: Dataset[tuple[Tensor, Tensor]]) -> Tensor:
    """
    Extracts targets from a dataset.

    Args:
        dataset (Dataset): Dataset of images and labels.

    Returns:
        Tensor: Tensor of targets.
    """
    return torch.tensor([el[1] for el in dataset])


def get_mnist_dataloaders(
        batch_size_train: int = 64,
        batch_size_test: int = 1000,
        **train_kwargs: Any
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """
    Returns MNIST data loaders.

    Args:
        batch_size_train (int): Batch size for training. Default is 64.
        batch_size_test (int): Batch size for testing. Default is 1000.
        **train_kwargs (Any): Additional arguments for DataLoader.

    Returns:
        tuple: Training and testing data loaders.
    """
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        FlattenTransform(),
    ])
    train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, **train_kwargs)
    test_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader


def get_cifar_dataloaders(
        batch_size_train: int = 128,
        batch_size_test: int = 1000,
        **train_kwargs: Any
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """
    Returns CIFAR-10 data loaders with normalization.

    Args:
        batch_size_train (int): Batch size for training. Default is 128.
        batch_size_test (int): Batch size for testing. Default is 1000.
        **train_kwargs (Any): Additional arguments for DataLoader.

    Returns:
        tuple: Training and testing data loaders.
    """
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

    return train_loader, test_loader


def get_dataloaders(name: str, **kwargs: Any) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """
    Returns data loaders for the specified dataset.

    Args:
        name (str): Dataset name ("MNIST" or "CIFAR").
        **kwargs (Any): Additional arguments for the dataset loaders.

    Returns:
        tuple: Training and testing data loaders.
    """
    if name == "MNIST":
        return get_mnist_dataloaders(**kwargs)
    elif name == "CIFAR":
        return get_cifar_dataloaders(**kwargs)
    else:
        raise AssertionError(f"Unsupported dataset: {name}")
