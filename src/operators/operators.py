import torch

from src.utils import check_tensor


def damped_harmonic_residual(pinn, t_i, zeta, omega, scale_factor):

    u = pinn(t_i)
    u_t = torch.autograd.grad(
        u, t_i, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_tt = torch.autograd.grad(
        u_t, t_i, torch.ones_like(u_t), create_graph=True, retain_graph=True
    )[0]

    u_t = u_t / scale_factor
    u_tt = u_tt / (scale_factor**2)

    return u_tt + 2 * zeta * omega * u_t + (omega**2) * u


def elastic_1d_residual(
    pinn: torch.nn.Module,
    x: torch.Tensor,
    F: float | torch.Tensor,
    A: float | torch.Tensor,
    E: float | torch.Tensor,
    f0: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    N = x.shape[0]
    device = x.device
    F = check_tensor(F, N, device)
    A = check_tensor(A, N, device)

    u = pinn(torch.cat([x, F, A], dim=-1))
    u_x = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]

    residual = E * A * u_xx + f0
    return residual, u, u_x, u_xx


OPERATOR_REGISTRY = {
    "damped_harmonic_oscillator": damped_harmonic_residual,
    "elastic_1d_residual": elastic_1d_residual,
}
