import torch


def linear_momentum_balance(
    grad_P: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:

    return grad_P + b