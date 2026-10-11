import torch


def mse(x_pred: torch.Tensor, x_gt: torch.Tensor, mean: bool = True) -> torch.Tensor:
    diff_sq = (x_pred - x_gt) ** 2
    return torch.mean(diff_sq) if mean else torch.sum(diff_sq)


def data_loss(
    u_pred: torch.Tensor,
    u_exact: torch.Tensor,
    psi_pred: torch.Tensor,
    psi_exact: torch.Tensor,
) -> torch.Tensor:
    l_u = mse(u_pred, u_exact)
    psi_scale = torch.mean(psi_exact**2) + 1e-8
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
) -> torch.Tensor:
    trac_ext_avg = 0.5 * (trac_ext[:-1] + trac_ext[1:])
    trac_ext = 0.5 * (trac_ext[:-1] + trac_ext[1:])
    t_pred = torch.einsum("eij, j -> ei", P_boundary, n_normal)
    trac_scale = torch.mean(trac_ext_avg**2) + 1e-8
    return mse(t_pred, trac_ext_avg) / trac_scale


def pako_loss(residual: torch.Tensor, F_ext: torch.Tensor) -> torch.Tensor:
    f_scale = torch.mean(F_ext**2) + 1.0
    rel_residual = residual / torch.sqrt(f_scale)
    return mse(rel_residual, torch.zeros_like(rel_residual))


def rl2e(x_pred: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    error_norm = torch.linalg.norm((x_pred - x_gt).flatten(), ord=2)
    gt_norm = torch.linalg.norm(x_gt.flatten(), ord=2) + eps
    return error_norm / gt_norm
