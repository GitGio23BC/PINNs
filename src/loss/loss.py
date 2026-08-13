import torch


def ic_loss(u_pred: torch.Tensor, u_exact: torch.Tensor) -> torch.Tensor:
    return torch.mean((u_pred - u_exact) ** 2)


def bc_loss(u_bc_pred: torch.Tensor, g_bc: torch.Tensor) -> torch.Tensor:
    return torch.mean((u_bc_pred - g_bc) ** 2)


def r_loss(n: torch.Tensor) -> torch.Tensor:
    return (n**2).mean()


def wr_loss(
    time_residuals: list[torch.Tensor], eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    L_r_stack = torch.stack(time_residuals)

    zeros = torch.zeros(1, device=L_r_stack.device)
    past_loss = torch.cumsum(torch.cat([zeros, L_r_stack[:-1]]), dim=0)
    weights = torch.exp(-eps * past_loss).detach()

    loss = torch.mean(weights * L_r_stack)

    return loss, torch.min(weights)
