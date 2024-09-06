import torch
from torch import Tensor

import numpy as np
import math

from .utils import safe_sqrt, safe_log, safe_div


def e_step(x: Tensor, sigma1: Tensor, sigma2: Tensor, p: Tensor) -> Tensor:
    log_p1 = safe_log(p) - safe_log(sigma1) - 0.5 * safe_div(x, sigma1).pow(2)
    log_p2 = safe_log(1 - p) - safe_log(sigma2) - 0.5 * safe_div(x, sigma2).pow(2)

    m = torch.where(log_p1 > log_p2, log_p1, log_p2)

    log_p1 -= m
    log_p2 -= m

    return log_p1.exp() / (log_p1.exp() + log_p2.exp())


def m_step(x: Tensor, pi: Tensor, dims: list[int]) -> tuple[Tensor, Tensor, Tensor]:
    n1 = pi.sum(dims, keepdims=True)
    n2 = math.prod(pi.shape[i] for i in dims) - n1
    sigma1 = safe_sqrt(safe_div((pi * x.pow(2)).sum(dims, keepdims=True), n1))
    sigma2 = safe_sqrt(safe_div(((1 - pi) * x.pow(2)).sum(dims, keepdims=True), n2))
    p = pi.mean(dims, keepdims=True)

    return sigma1, sigma2, p


def em_gaussian_mixture(x: Tensor, dims: list[int], n_iter: int = 1000, *, pi : Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if pi is None:
        sigma = safe_sqrt(x.pow(2).mean(dims, keepdims=True))

        sigma1 = sigma / 2
        sigma2 = sigma * 2
        p = torch.zeros_like(x) + 0.5
    else:
        sigma1, sigma2, p = m_step(x, pi, dims)

    for _ in range(n_iter):
        pi = e_step(x, sigma1, sigma2, p)
        sigma1, sigma2, p = m_step(x, pi, dims)

    return pi, sigma1, sigma2, p
