import torch


def ic_loss(u_pred: torch.Tensor, u_exact: torch.Tensor) -> torch.Tensor:
    return mse(u_pred, u_exact)


def bc_loss(u_bc_pred: torch.Tensor, g_bc: torch.Tensor) -> torch.Tensor:
    return mse(u_bc_pred, g_bc)


def r_loss(n: torch.Tensor) -> torch.Tensor:
    return mse(n, torch.zeros_like(n))


def mse(x_in: torch.Tensor, x_gt: torch.Tensor, mean: bool = True) -> torch.Tensor:
    mse_tensor = (x_in - x_gt) ** 2
    if mean:
        mse_tensor = torch.mean(mse_tensor)
    return mse_tensor


def rmse(x_in: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.linalg.norm(x_in - x_gt, ord=2)
    dnum = torch.linalg.norm(x_gt, ord=2) + eps

    return num / dnum


def wr_loss(
    time_residuals: list[torch.Tensor], eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    L_r_stack = torch.stack(time_residuals)

    zeros = torch.zeros(1, device=L_r_stack.device)
    past_loss = torch.cumsum(torch.cat([zeros, L_r_stack[:-1]]), dim=0)
    weights = torch.exp(-eps * past_loss).detach()

    loss = torch.mean(weights * L_r_stack)

    return loss, torch.min(weights)
