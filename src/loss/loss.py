import torch


def mse(x_pred: torch.Tensor, x_gt: torch.Tensor, mean: bool = True) -> torch.Tensor:

    diff_sq = (x_pred - x_gt) ** 2
    return torch.mean(diff_sq) if mean else diff_sq


def ic_loss(u_pred: torch.Tensor, u_exact: torch.Tensor) -> torch.Tensor:
    return mse(u_pred, u_exact)


def bc_loss(u_bc_pred: torch.Tensor, g_bc: torch.Tensor) -> torch.Tensor:
    return mse(u_bc_pred, g_bc)


def r_loss(residual: torch.Tensor) -> torch.Tensor:
    return mse(residual, torch.zeros_like(residual))


def traction_bc_loss(
    P: torch.Tensor, n_normal: torch.Tensor, g_traction: torch.Tensor
) -> torch.Tensor:
    traction_pred = (P @ n_normal.unsqueeze(-1)).squeeze(-1)
    return mse(traction_pred, g_traction)


def rl2e(
    x_pred: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:

    error_norm = torch.linalg.norm((x_pred - x_gt).flatten(), ord=2)
    gt_norm = torch.linalg.norm(x_gt.flatten(), ord=2) + eps
    return error_norm / gt_norm
