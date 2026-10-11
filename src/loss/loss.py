import torch


def mse(x_pred: torch.Tensor, x_gt: torch.Tensor, mean: bool = True) -> torch.Tensor:

    diff_sq = (x_pred - x_gt) ** 2
    return torch.mean(diff_sq) if mean else torch.sum(diff_sq)

def data_loss(u_pred: torch.Tensor, u_exact: torch.Tensor, psi_pred: torch.Tensor, psi_exact: torch.Tensor) -> torch.Tensor:
    l_u = mse(u_pred, u_exact)
    l_psi = mse(psi_pred, psi_exact)
    return l_u + l_psi

def dirichlet_loss(u_bc_pred: torch.Tensor) -> torch.Tensor:
    return mse(u_bc_pred, torch.zeros_like(u_bc_pred))

def neumann_loss(P: torch.Tensor, n_normal: torch.Tensor, F_ext: torch.Tensor) -> torch.Tensor:
    scal= torch.sum(P * n_normal, dim=-1)
    return mse(scal, F_ext)    

def pako_loss(div_P: torch.Tensor, b: torch.Tensor, rho: float, u_tt: torch.Tensor) -> torch.Tensor:
    residual= div_P - b - rho * u_tt
    return mse(residual, torch.zeros_like(residual))



def rl2e(x_pred: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    error_norm = torch.linalg.norm((x_pred - x_gt).flatten(), ord=2)
    gt_norm = torch.linalg.norm(x_gt.flatten(), ord=2) + eps
    return error_norm / gt_norm
