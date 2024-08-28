from torch import Tensor


def safe_sqrt(x: Tensor) -> Tensor:
    return (x + 1e-30).sqrt()


def safe_log(x: Tensor) -> Tensor:
    return (x + 1e-30).log()


def safe_div(x: Tensor, y: Tensor) -> Tensor:
    return x / (y + 1e-15)
