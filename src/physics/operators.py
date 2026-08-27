import torch

from src.utils import div, grad, voigt_tensor

from .constitutive import Material
from .neo_hookean import (
    FungEnergy_1D,
    FungEnergy_2D,
    Viscoelastic,
    linear_momentum_balance,
)


def haslach_constitutive_evolution_1D(
    pinn: torch.nn.Module,
    x: torch.Tensor,
    k: float,
    c1: float,
    c: float,
    b: torch.Tensor,
    S: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    x.requires_grad_(True)

    u = pinn(x)
    grad_u = grad(u, x)

    body = Material(grad_u, S)

    E = body.E
    P = body.P
    div_P = div(P, x)

    costitutive_eq = Viscoelastic(FungEnergy_1D(c, c1), k)
    haslach_residue = - costitutive_eq.haslach_equation(E, S)

    momentum_residue = - linear_momentum_balance(div_P, b)

    return u, haslach_residue, momentum_residue


def haslach_constitutive_evolution_2D(
    pinn: torch.nn.Module,
    x: torch.Tensor,
    k: float,
    c1: float,
    c2: float,
    c3: float,
    c: float,
    b: torch.Tensor,
    S: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    x.requires_grad_(True)
    if S.ndim == 3 and S.shape[-1] == 2:
        S_tensor = S
        S_vec = voigt_tensor(S, is_shear=False).unsqueeze(-1)
    elif S.ndim == 2 and S.shape[-1] == 3:
        S_vec = S.unsqueeze(-1)
        S_tensor = torch.stack(
            [
                torch.stack([S[:, 0], S[:, 2]], dim=-1),
                torch.stack([S[:, 2], S[:, 1]], dim=-1),
            ],
            dim=1,
        )
    else:
        raise ValueError()

    u = pinn(x)
    grad_u = grad(u, x)

    body = Material(grad_u, S_tensor)

    E = body.E
    P = body.P
    div_P = div(P, x)

    costitutive_eq = Viscoelastic(FungEnergy_2D(c, c1, c2, c3), k)
    haslach_residue = -costitutive_eq.haslach_equation(E, S_vec)

    momentum_residue = -linear_momentum_balance(div_P, b)

    return u, haslach_residue, momentum_residue


OPERATOR_REGISTRY = {
    "viscoelastic_residual_Fung_1D": haslach_constitutive_evolution_1D,
    "viscoelastic_residual_Fung_2D": haslach_constitutive_evolution_2D,
}
