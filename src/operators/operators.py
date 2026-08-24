import torch


def haslach_constitutive_evolution_1D(
    pinn: torch.nn.Module,
    t: torch.Tensor,
    k: float,
    ct: float,
    c: float,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    t.requires_grad_(True)

    E = pinn(t)
    E_t = torch.autograd.grad(
        E,
        t,
        grad_outputs=torch.ones_like(E),
        create_graph=True,
        retain_graph=True,
    )[0]

    exp_term = torch.exp(
        ct * E**2
    )  # Note, this is only the exponential term of the psi
    psi_E = 2.0 * c * ct * E * exp_term  # dψ/dE

    H = 2.0 * c * ct * (1.0 + 2.0 * ct * E**2) * exp_term  # d²ψ/dE²

    H_inv2 = 1.0 / (H**2)

    residual = E_t - (-k * H_inv2 * (psi_E - y))
    return residual, E


import torch


def haslach_constitutive_evolution_2D(
    pinn: torch.nn.Module,
    t: torch.Tensor,
    k: float,
    c1: float,
    c2: float,
    c3: float,
    c: float,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    t.requires_grad_(True)

    # E shape: (N, 2)
    E = pinn(t)

    # Compute time derivative dE/dt via autograd
    grad_E1, grad_E2 = torch.autograd.grad(
        outputs=(E[:, 0], E[:, 1]),
        inputs=(t, t),
        grad_outputs=(torch.ones_like(E[:, 0]), torch.ones_like(E[:, 1])),
        create_graph=True,
        retain_graph=True,
    )

    E_t = torch.cat([grad_E1, grad_E2], dim=-1)  # Shape: (N, 2)

    Q = torch.tensor([[c1, 0.5 * c3], [0.5 * c3, c2]], dtype=t.dtype, device=t.device)
    B = 2.0 * Q  # Shape: (2, 2) # Symmetic Q so Q+Q.T

    # Quadratic scalar per sample: s = E^T @ Q @ E -> Shape: (N, 1)
    s = torch.einsum("bi,ij,bj->b", E, Q, E).unsqueeze(-1)
    exp_s = c * torch.exp(s)  # Shape: (N, 1)

    BE = torch.matmul(E, B)  # Shape: (N, 2)
    grad_psi = exp_s * BE  # Shape: (N, 2)

    # Hessian matrix per sample: H = c * exp(s) * [ B + BE ⊗ BE ]
    # Shape: (N, 2, 1) x (N, 1, 2) -> (N, 2, 2)
    BE_outer = torch.bmm(BE.unsqueeze(-1), BE.unsqueeze(1))
    B_expanded = B.unsqueeze(0).expand_as(BE_outer)  # Shape: (N, 2, 2)

    H = exp_s.unsqueeze(-1) * (B_expanded + BE_outer)  # Shape: (N, 2, 2)

    H_inv = torch.linalg.inv(H)
    H_inv2 = torch.bmm(H_inv, H_inv)  # Shape: (N, 2, 2)

    if y.ndim == 1:
        y = y.unsqueeze(0)  # Shape: (1, 2)
    driving_force = (grad_psi - y).unsqueeze(-1)  # Shape: (N, 2, 1)

    # Residual: E_t - (-k * H^-2 @ (grad_psi - y))
    rhs = -k * torch.bmm(H_inv2, driving_force).squeeze(-1)  # Shape: (N, 2)
    residual = E_t - rhs

    return residual, E


OPERATOR_REGISTRY = {
    "viscoelastic_residual_Fung_1D": haslach_constitutive_evolution_1D,
    "viscoelastic_residual_Fung_2D": haslach_constitutive_evolution_2D,
}
