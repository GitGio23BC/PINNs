import torch
from .constitutive import StrainEnergy

class Ogden:
    def __init__(
            self,
            mu: float,
            alpha: float,
            beta: float,
            J: torch.Tensor,     
    ):
        self.mu = mu #Hidden layer
        self.alpha = alpha #Hidden layer
        self.beta = beta #Hidden layer
        self.J = J
        self.g = beta**(-2)*(beta*torch.log(J)+J**(-beta)-1)
        self.Jg_J = -(J**(-beta)-1)/(beta)
        self.Jg_JJ = J**(-beta-1)

    def energy(self, psi, E_voigt=None):
        self.psi = psi
        return psi 
    
    def grad(self, E_voigt):
        return torch.autograd.grad(self.psi, E_voigt, grad_outputs=torch.ones_like(E_voigt), create_graph=True)[0]

    def hessian(self, E_voigt=None):
        pass

    def hills_constitutive_inequality(self):
        return self.mu*self.alpha>0, self.beta>0, self.J*self.Jg_JJ>0


class Kinematics:
    def __init__(self, grad_u: torch.Tensor) -> None:
        self.grad_u = grad_u
        dim = grad_u.shape[-1]
        self.I = torch.eye(dim, device=grad_u.device, dtype=grad_u.dtype).unsqueeze(0)
        self.F = self.I + self.grad_u
        self.C = self.F.mT @ self.F
        self.E = 0.5 * (self.C - self.I)
        self.J = torch.linalg.det(self.F)

    def compute_P(self, S: torch.Tensor):
        return self.F @ S


class HolzapfelEnergy_2D(StrainEnergy):
    def __init__(
        self,
        c: float,
        c1: float,
        c2: float,
        c3: float,
        device: torch.device | str = "cpu",
    ) -> None:

        self.c = c
        self.Q = torch.tensor(
            [
                [c1, 0.5 * c3, 0.0],
                [0.5 * c3, c2, 0.0],
                [0.0, 0.0, 0.25 * (c1 + c2)],
            ],
            dtype=torch.float32,
            device=device,
        )

    def energy(self, E_voigt: torch.Tensor) -> torch.Tensor:
        E_vec = E_voigt.unsqueeze(-1) if E_voigt.ndim == 2 else E_voigt

        s = E_vec.mT @ self.Q @ E_vec

        return self.c * (torch.exp(s) - 1.0)

    def grad(self, E_voigt: torch.Tensor) -> torch.Tensor:
        E_vec = E_voigt.unsqueeze(-1) if E_voigt.ndim == 2 else E_voigt

        B = 2 * self.Q  # self.Q + self.Q.mT
        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)
        S_vec = self.c * exp_s * (B @ E_vec)

        return S_vec.squeeze(-1) if E_voigt.ndim == 2 else S_vec

    def hessian(self, E_voigt: torch.Tensor) -> torch.Tensor:
        E_vec = E_voigt.unsqueeze(-1) if E_voigt.ndim == 2 else E_voigt

        B = 2 * self.Q
        s = E_vec.mT @ self.Q @ E_vec
        exp_s = torch.exp(s)

        BE = B @ E_vec
        BE_outer = BE @ BE.mT

        return self.c * exp_s * (B.unsqueeze(0) + BE_outer)


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

    def __init__(
        self,
        psi: StrainEnergy,
        k_relax: float,
        eps_reg: float = 1e-6,
    ) -> None:
        self.psi = psi
        self.k_relax = k_relax
        self.eps_reg = eps_reg

    def haslach_equation(
        self,
        E_voigt: torch.Tensor,
        S_applied_voigt: torch.Tensor,
    ) -> torch.Tensor:
        S_int = self.psi.grad(E_voigt)
        H = self.psi.hessian(E_voigt)

        dS = (S_int - S_applied_voigt).unsqueeze(-1)
        reg = self.eps_reg * torch.eye(3, device=H.device, dtype=H.dtype)
        H_reg = H + reg

        iHdS = torch.linalg.solve(H_reg, dS)
        iiHdS = torch.linalg.solve(H_reg, iHdS)

        E_dot_target = -self.k_relax * iiHdS

        return E_dot_target.squeeze(-1)
