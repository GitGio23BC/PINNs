import torch


def check_tensor(
    x: float | torch.Tensor, N: int, device: torch.device | str
) -> torch.Tensor:
    if isinstance(x, (float, int)):
        return torch.full((N, 1), float(x), device=device, dtype=torch.float32)

    if isinstance(x, torch.Tensor):
        if x.dim() == 0 or (x.dim() == 1 and x.numel() == 1):
            return x.expand(N, 1).to(device)
        if x.dim() == 1 and x.shape[0] == N:
            return x.unsqueeze(-1).to(device)
        return x.to(device)

    raise TypeError(f"Unsupported input type for tensor conversion: {type(x)}")
