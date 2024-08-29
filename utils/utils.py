import torch
from torch import Tensor


def safe_sqrt(x: Tensor) -> Tensor:
    return (x + 1e-30).sqrt()


def safe_log(x: Tensor) -> Tensor:
    return (x + 1e-30).log()


def safe_div(x: Tensor, y: Tensor) -> Tensor:
    return x / (y + 1e-15)


def t_log_likelihood(data: Tensor, nu: Tensor) -> Tensor:
    ll = torch.lgamma((nu + 1) / 2).mean() * data.numel()
    ll -= 0.5 * (torch.pi * nu).log().mean() * data.numel()
    ll -= torch.lgamma(nu / 2).mean() * data.numel()
    ll -= (0.5 * (nu + 1) * (1 + data.pow(2) / nu).log()).sum()
    return ll
