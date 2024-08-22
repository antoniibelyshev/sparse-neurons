from torch import Tensor


eps = 1e-20


def safe_sqrt(x: Tensor) -> Tensor:
    return (x + eps).sqrt()


def safe_log(x: Tensor) -> Tensor:
    return (x + eps).log()
