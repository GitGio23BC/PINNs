import torch


class Kinematics:
    def __init__(self, grad_u: torch.Tensor) -> None:
        self.grad_u = grad_u
        dim = grad_u.shape[-1]
        self.I = torch.eye(dim, device=grad_u.device, dtype=grad_u.dtype).unsqueeze(0)
        self.F = self.I + self.grad_u
        self.C = self.F.mT @ self.F
        self.E = 0.5 * (self.C - self.I)
        self.J = torch.linalg.det(self.F).clamp(min=1e-8)

    def compute_P(self, S: torch.Tensor) -> torch.Tensor:
        return self.F @ S

    def compute_tau(self, S: torch.Tensor) -> torch.Tensor:
        return self.F @ S @ self.F.mT

    def compute_sigma(self, S: torch.Tensor) -> torch.Tensor:
        tau = self.compute_tau(S)
        return tau / self.J.unsqueeze(-1).unsqueeze(-1)


class Ogden:
    def __init__(
        self,
        psi: torch.Tensor,
        mus: torch.Tensor,
        alphas: torch.Tensor,
        beta: float,
        lam: float,
        kin: Kinematics,
    ):
        self.psi = psi  # Model output
        self.mus = mus  # Hidden layer
        self.alphas = alphas  # Hidden layer
        self.beta = beta  # Hidden layer
        self.lam = lam
        self.kin = kin
        self.g, self.Jg_J, self.Jg_JJ = self.get_g()

    def get_g(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.g = self.beta ** (-2) * (
            self.beta * torch.log(self.kin.J) + self.kin.J ** (-self.beta) - 1.0
        )
        self.Jg_J = -(self.kin.J ** (-self.beta) - 1) / (self.beta)
        self.Jg_JJ = self.kin.J ** (-self.beta - 1)

        return self.g, self.Jg_J, self.Jg_JJ

    def get_principal_stretches(self, eps: float = 1e-12) -> torch.Tensor:
        # Compute the eigenvalues using analytic formula
        tr_C = self.kin.C[..., 0, 0] + self.kin.C[..., 1, 1]
        det_C = (
            self.kin.C[..., 0, 0] * self.kin.C[..., 1, 1]
            - self.kin.C[..., 0, 1] * self.kin.C[..., 1, 0]
        )

        delta = torch.clamp((tr_C**2) - 4.0 * det_C, min=0.0)
        sqrt_delta = torch.sqrt(delta + eps)

        eig_1 = 0.5 * (tr_C + sqrt_delta)
        eig_2 = 0.5 * (tr_C - sqrt_delta)

        lam_1 = torch.sqrt(torch.clamp(eig_1, min=1e-8))
        lam_2 = torch.sqrt(torch.clamp(eig_2, min=1e-8))
        lam_2D = torch.stack([lam_1, lam_2], dim=-1)

        ones = torch.ones_like(lam_2D[..., :1])
        return torch.cat([lam_2D, ones], dim=-1)

    def get_2Dpsi(self) -> torch.Tensor:

        lam_princ = self.get_principal_stretches()

        lam_pow = lam_princ.unsqueeze(-1) ** self.alphas
        phi = (torch.sum(lam_pow, dim=-2) - 3.0) / self.alphas
        psi_deviatronic = torch.sum(
            self.mus * (phi - torch.log(self.kin.J).unsqueeze(-1)), dim=-1
        )
        psi_volumetric = self.lam * self.get_g()[0]

        psi = psi_deviatronic + psi_volumetric

        return psi

    def get_C_SE(self):
        pass

    def get_S(self, E: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            self.psi, E, grad_outputs=torch.ones_like(E), create_graph=True
        )[0]

    def hills_constitutive_inequality(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.all(self.mus * self.alphas > 0),
            torch.tensor(self.beta > 0),
            torch.all(self.kin.J * self.Jg_JJ > 0),
        )
