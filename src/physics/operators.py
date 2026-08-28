import torch

from src.utils import div, grad, voigt_to_tensor

from .constitutive import Material
from .equations import (
    HUGO,
    FungEnergy_1D,
    FungEnergy_2D,
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

    costitutive_eq = HUGO(FungEnergy_1D(c, c1), k)
    haslach_residue = -costitutive_eq.haslach_equation(E, S)

    momentum_residue = -linear_momentum_balance(div_P, b)

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    x.requires_grad_(True)

    u = pinn(x)
    grad_u = grad(u, x)

    energy_func = FungEnergy_2D(c, c1, c2, c3)
    costitutive_eq = HUGO(energy_func, k)

    body = Material(grad_u)
    E = body.E

    S_vec = energy_func.grad(E)
    S = voigt_to_tensor(S_vec)
    P = body.compute_P(S)
    div_P = div(P, x)

    haslach_residue = torch.zeros(S_vec.size()) #-costitutive_eq.haslach_equation(E, S_vec)

    momentum_residue = -linear_momentum_balance(div_P, b)

    return u, haslach_residue, momentum_residue


OPERATOR_REGISTRY = {
    "viscoelastic_residual_Fung_1D": haslach_constitutive_evolution_1D,
    "viscoelastic_residual_Fung_2D": haslach_constitutive_evolution_2D,
}
