import torch

from src.utils import voigt_tensor
from .constitutive import STrainEnergy


class FungEnergy_1D(STrainEnergy):
    def __init__(self, c: float, c1: float) -> None:
        self.c = c
        self.c1 = c1

    def energy(self, E: torch.Tensor) -> torch.Tensor:
        exp_E = torch.exp(self.c1 * E**2)
        return self.c * (exp_E - 1.0)

    def grad(self, E: torch.Tensor) -> torch.Tensor:
        exp_E = torch.exp(self.c1 * E**2)
        return 2.0 * self.c * self.c1 * E * exp_E

    def hessian(self, E: torch.Tensor) -> torch.Tensor:
        exp_E = torch.exp(self.c1 * E**2)
        return 2.0 * self.c * self.c1 * (1.0 + 2.0 * self.c1 * E**2) * exp_E


class FungEnergy_2D(STrainEnergy):
    def __init__(self, c: float, c1: float, c2: float, c3: float) -> None:
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

        E_vec = voigt_tensor(E, is_shear=True).unsqueeze(-1)

        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)
        return self.c * (exp_s - 1.0)

    def grad(self, E: torch.Tensor) -> torch.Tensor:
        self.Q = self.Q.to(device=E.device, dtype=E.dtype)
        E_vec = voigt_tensor(E, is_shear=True).unsqueeze(-1)

        B = 2.0 * self.Q
        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)
        return self.c * exp_s * (B @ E_vec)

    def hessian(self, E: torch.Tensor) -> torch.Tensor:
        self.Q = self.Q.to(device=E.device, dtype=E.dtype)
        E_vec = voigt_tensor(E, is_shear=True).unsqueeze(-1)

        B = 2.0 * self.Q
        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)

        BE = B @ E_vec
        BE_outer = BE @ BE.mT
        return exp_s * (B.expand_as(BE_outer) + BE_outer)


class HUGO:
    def __init__(self, psi: STrainEnergy, k: float) -> None:
        self.psi = psi
        self.k = k

    def direct_haslach(self, E: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        psi_E = self.psi.grad(E)
        H = self.psi.hessian(E)

        H_inv = torch.linalg.inv(H) if H.shape[-1] != 1 else 1.0 / H
        H_inv2 = H_inv @ H_inv
        # REMEMBER TO OPTIMISE THE MATRICS MOLTIPLOCATION

        return -self.k * (H_inv2 @ (psi_E - S))

    def inverse_haslach(self, E: torch.Tensor, E_t: torch.Tensor) -> torch.Tensor:

        psi_E = self.psi.grad(E)
        H = self.psi.hessian(E)

        E_t_vec = voigt_tensor(E_t, is_shear=True).unsqueeze(-1)

        H2 = H @ H
        viscous_stress = (H2 @ E_t_vec) / self.k

        S_vec = viscous_stress + psi_E
        return S_vec.squeeze(-1)
