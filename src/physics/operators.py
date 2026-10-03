import torch

from ..utils import div
from .equations import Kinematics


def pako_residual_2D(
    kin: Kinematics,
    S: torch.Tensor,
    X_ref: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:

    P = kin.compute_P(S)
    div_P = div(P, X_ref)

    return div_P + b

