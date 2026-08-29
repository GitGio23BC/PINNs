import torch

from src.utils import d_dt, div, grad, voigt_tensor, voigt_to_tensor

from .constitutive import Material
from .equations import (
    HUGO,
    FungEnergy_1D,
    FungEnergy_2D,
)


def haslach_constitutive_evolution_1D(
    pinn: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    f: torch.Tensor,
    k: float,
    c1: float,
    c: float,
    b: torch.Tensor,
    S: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    x.requires_grad_(True)
    t.requires_grad_(True)
    f.requires_grad_(True)

    u = pinn(torch.cat([x, t, f]))
    grad_u = grad(u, x)

    energy_func = FungEnergy_1D(c, c1)
    costitutive_eq = HUGO(energy_func, k)

    body = Material(grad_u)
    E = body.E
    E_t = d_dt(E, t)

    Ev = voigt_tensor(E)
    Ev_t = voigt_tensor(E_t)

    Sv = costitutive_eq.inverse_haslach(Ev, Ev_t)
    S = voigt_to_tensor(Sv)

    P = body.compute_P(S)
    div_P = div(P, x)

    momentum_residue = div_P + b

    return u, P, momentum_residue

def haslach_constitutive_evolution_2D(
    pinn: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    f: torch.Tensor,
    k: float,
    c1: float,
    c2: float,
    c3: float,
    c: float,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    x.requires_grad_(True)
    t.requires_grad_(True)
    f.requires_grad_(True)

    u = pinn(x, t, f)
    grad_u = grad(u, x)

    energy_func = FungEnergy_2D(c, c1, c2, c3)
    constitutive_eq = HUGO(energy_func, k)

    body = Material(grad_u)
    E = body.E
    E_t = d_dt(E, t)

    Sv = constitutive_eq.inverse_haslach(E, E_t)
    S = voigt_to_tensor(Sv)

    P = body.compute_P(S)
    div_P = div(P, x)

    momentum_residual = div_P + b

    return u, P, momentum_residual


OPERATOR_REGISTRY = {
    "viscoelastic_residual_Fung_1D": haslach_constitutive_evolution_1D,
    "viscoelastic_residual_Fung_2D": haslach_constitutive_evolution_2D,
}
