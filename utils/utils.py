from torch import Tensor


def safe_sqrt(x: Tensor) -> Tensor:
    """
    Returns the element-wise square root of `x`, avoiding negative values.
    
    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Square root of `x`.
    """
    return (x + 1e-20).sqrt()


def safe_log(x: Tensor) -> Tensor:
    """
    Returns the element-wise natural log of `x`, avoiding zero/negative inputs.
    
    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Logarithm of `x`.
    """
    return (x + 1e-20).log()


def safe_div(x: Tensor, y: Tensor) -> Tensor:
    """
    Element-wise division of `x` by `y`, avoiding division by zero.
    
    Args:
        x (Tensor): Numerator tensor.
        y (Tensor): Denominator tensor.

    Returns:
        Tensor: Result of `x / y`.
    """
    return x / (y + 1e-10)
