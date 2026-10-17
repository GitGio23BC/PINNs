import torch


def mse(x_pred: torch.Tensor, x_gt: torch.Tensor, mean: bool = True) -> torch.Tensor:
    diff_sq = (x_pred - x_gt) ** 2
    return torch.mean(diff_sq) if mean else torch.sum(diff_sq)


def data_loss(
    u_pred: torch.Tensor,
    u_exact: torch.Tensor,
    psi_pred: torch.Tensor,
    psi_exact: torch.Tensor,
    u_scale: torch.Tensor,
    psi_scale: torch.Tensor,
) -> torch.Tensor:
    l_u = mse(u_pred, u_exact) / u_scale
    l_psi = mse(psi_pred, psi_exact) / psi_scale
    return l_u + l_psi


def dirichlet_loss(
    u_pred: torch.Tensor,
    fixed_dofs: torch.Tensor,
    u_fixed_gt: torch.Tensor,
) -> torch.Tensor:
    u_flat = u_pred.reshape(-1)
    u_fixed_pred = u_flat[fixed_dofs]
    u_fixed_gt = u_fixed_gt.reshape(-1)
    return mse(u_fixed_pred, u_fixed_gt)


def neumann_loss(
    P_boundary: torch.Tensor,
    n_normal: torch.Tensor,
    trac_ext: torch.Tensor,
    sigma_ref: float = 1.5e5,  # Modulo di elasticità/stress di riferimento tipico (es. 100 kPa)
) -> torch.Tensor:
    if trac_ext.shape[0] != P_boundary.shape[0]:
        trac_ext_avg = 0.5 * (trac_ext[:-1] + trac_ext[1:])
    else:
        trac_ext_avg = trac_ext
    t_pred = torch.einsum("eij, j -> ei", P_boundary, n_normal)
    return mse(t_pred, trac_ext_avg) / sigma_ref**2 #da valutare perchè normalizzare con la sigma


def pako_loss(
    residual: torch.Tensor,
    force_scale: torch.Tensor,
    free_dofs: torch.Tensor,
) -> torch.Tensor:
    residual_free = residual.flatten()[free_dofs]
    if residual_free.numel() == 0:
        raise ValueError("free_dofs must select at least one residual DOF.")
    return torch.mean(residual_free**2) / force_scale


def rl2e(x_pred: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    error_norm = torch.linalg.norm((x_pred - x_gt).flatten(), ord=2)
    gt_norm = torch.linalg.norm(x_gt.flatten(), ord=2) + eps
    return error_norm / gt_norm
