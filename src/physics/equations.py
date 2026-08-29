import torch
from src.utils import voigt_tensor
from .constitutive import STrainEnergy

def linear_momentum_balance(
    div_P: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:

    return div_P + b


class FungEnergy_2D(STrainEnergy):
    def __init__(
        self,
        c: float,
        c1: float,
        c2: float,
        c3: float,
    ) -> None:

        self.c = c
        self.Q = torch.tensor(
            [
                [c1, 0.5 * c3, 0.0],
                [0.5 * c3, c2, 0.0],
                [0.0, 0.0, 0.25 * (c1 + c2)],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

    def energy(self, E: torch.Tensor) -> torch.Tensor:
        self.Q = self.Q.to(device=E.device, dtype=E.dtype)
        E_vec = voigt_tensor(E).unsqueeze(-1)

        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)

        return self.c * (exp_s - 1.0)

    def grad(self, E: torch.Tensor) -> torch.Tensor:
        self.Q = self.Q.to(device=E.device, dtype=E.dtype)
        E_vec = voigt_tensor(E).unsqueeze(-1)

        B = 2 * self.Q  # self.Q + self.Q.mT
        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)

        return self.c * exp_s * (B @ E_vec)

    def hessian(self, E: torch.Tensor) -> torch.Tensor:
        self.Q = self.Q.to(device=E.device, dtype=E.dtype)
        E_vec = voigt_tensor(E).unsqueeze(-1)

        B = 2 * self.Q
        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)

        BE = B @ E_vec
        BE_outer = BE @ BE.mT

        return exp_s * (B.expand_as(BE_outer) + BE_outer)

class HUGO:
    """
    A Holzapfel-Gasser-Ogden (HGO) model.
    It implements a very basic version assuming
    isotropy.

    The material consists of:

        - 150 ml of Prosecco
        - 20 ml of lemon balm or elderflower syrup
        - seltzer or soda
        - 1 slice of lemon or lime
        - ice

    Remember to decor with mint leaves.
    """

    def __init__(self, psi: STrainEnergy, k: float) -> None:
        self.psi = psi
        self.k = k

    def haslach_equation(self, E: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        self.S = self.psi.grad(E)
        H = self.psi.hessian(E)

        H_inv = torch.linalg.inv(H) if H.shape[-1] != 1 else 1.0 / H
        H_inv2 = H_inv @ H_inv

        return -self.k * (H_inv2 @ (self.S - S))


class Material:
    def __init__(self, grad_u: torch.Tensor, S: torch.Tensor | None = None) -> None:
        self.grad_u = grad_u
        self.S = S

        d = grad_u.shape[-1]
        self.I = torch.eye(d, device=grad_u.device, dtype=grad_u.dtype).unsqueeze(0)

        self.F = self.I + self.grad_u
        self.C = self.F.mT @ self.F
        self.E = 0.5 * (self.C - self.I)
        self.J = torch.linalg.det(self.F)

        if self.S is not None:
            self.compute_P(self.S)
        
    def compute_P(self, S: torch.Tensor):
        self.S = S
        self.P = self.F @ self.S
        return self.P