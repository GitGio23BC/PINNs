import torch

from ..utils import div, grad, voigt_tensor, voigt_to_tensor
from .equations import HUGO, HolzapfelEnergy_2D, Kinematics


def haslach_constitutive_residual_2D(
    u_pred: torch.Tensor,
    S_pred: torch.Tensor,
    X_ref: torch.Tensor,
    E_prev: torch.Tensor,
    dt: float,
    k_relax: float,
    c: float,
    c1: float,
    c2: float,
    c3: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    grad_u = grad(u_pred, X_ref)
    kin = Kinematics(grad_u)

    E_curr_voigt = voigt_tensor(kin.E, is_shear=True)
    E_prev_voigt = voigt_tensor(E_prev) if E_prev.ndim == 3 else E_prev

    E_dot = (E_curr_voigt - E_prev_voigt) / dt

    psi = HolzapfelEnergy_2D(c=c, c1=c1, c2=c2, c3=c3, device=X_ref.device)
    visco_model = HUGO(psi=psi, k_relax=k_relax)

    S_curr_voigt = voigt_tensor(S_pred, is_shear=False) if S_pred.ndim == 3 else S_pred

    E_dot_pred = visco_model.haslach_equation(E_curr_voigt, S_curr_voigt)

    residual = E_dot - E_dot_pred

    return E_curr_voigt, residual


def pako_residual_2D(
    u_pred: torch.Tensor,
    S_pred: torch.Tensor,
    X_ref: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:

    grad_u = grad(u_pred, X_ref)
    kin = Kinematics(grad_u)
    S = (
        voigt_to_tensor(S_pred, is_shear=False)
        if S_pred.ndim == 2 and S_pred.shape[-1]
        else S_pred
    )

    P = kin.compute_P(S)
    div_P = div(P, X_ref)

    return div_P + b


OPERATOR_REGISTRY = {
    "viscoelastic_residual_Holzapfel_2D": haslach_constitutive_residual_2D,
    "piola_kirchhoff_quasi_static_residual_2D": pako_residual_2D,
}
