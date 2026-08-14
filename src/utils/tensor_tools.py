import torch


def check_tensor(x, N: int, device: torch.device) -> torch.Tensor:
    if isinstance(x, (float, int)):
        x_tensor = torch.full((N, 1), float(x), device=device)
    else:
        x_tensor = x
    return x_tensor
