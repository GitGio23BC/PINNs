import torch

from ..utils import d_dt, div, grad, voigt_tensor, voigt_to_tensor
from .equations import HUGO, Kinematics


def haslach_constitutive_residual_2D(
    u_pred: torch.Tensor,
    S_pred: torch.Tensor,
    X_ref: torch.Tensor,
    t: torch.Tensor,
    visco_model: HUGO,
) -> torch.Tensor:

    grad_u = grad(u_pred, X_ref)
    kin = Kinematics(grad_u)

    E_voigt = voigt_tensor(kin.E, is_shear=False)

    E_dot_voigt = d_dt(E_voigt, t)

    S_voigt = S_pred

    E_dot_pred = visco_model.haslach_equation(E_voigt, S_voigt)

    residual = E_dot_voigt - E_dot_pred

    return residual


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
