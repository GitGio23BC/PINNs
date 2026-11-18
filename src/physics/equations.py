import torch


class Kinematics:
    def __init__(self, grad_u: torch.Tensor) -> None:
        self.grad_u = grad_u
        dim = grad_u.shape[-1]
        self.I = torch.eye(dim, device=grad_u.device, dtype=grad_u.dtype).unsqueeze(0)
        self.F_base = self.I + self.grad_u

        self.E = 0.5 * (self.F_base.mT @ self.F_base - self.I)
        self.C = 2.0 * self.E + self.I
        self.J = torch.linalg.det(self.F_base)
        self.F = self.F_base

    #def compute_P(self: torch.Tensor) -> torch.Tensor:
    #    return torch.autograd.grad(
    #        outputs=self.psi.sum(),  # type: ignore
    #        inputs=self.kin.F,
    #        create_graph=True,
    #        retain_graph=True,
    #    )[0]

    def compute_tau(self, S: torch.Tensor) -> torch.Tensor:
        return self.F @ S @ self.F.mT

    def compute_sigma(self, S: torch.Tensor) -> torch.Tensor:
        tau = self.compute_tau(S)
        return tau / self.J.unsqueeze(-1).unsqueeze(-1)


class Ogden:
    def __init__(
        self,
        mus: torch.Tensor,
        alphas: torch.Tensor,
        beta: float,
        lam: float,
        kin: Kinematics,
    ):
        self.mus = mus  # Hidden layer
        self.alphas = alphas  # Hidden layer
        self.beta = beta  # Hidden layer
        self.lam = lam  # Hidden layer
        self.kin = kin
        self.psi = None

    def get_g(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.beta ** (-2.0) * (
            self.beta * torch.log(self.kin.J) + self.kin.J ** (-self.beta) - 1.0
        )
        Jg_J = -(self.kin.J ** (-self.beta) - 1.0) / self.beta
        Jg_JJ = self.kin.J ** (-self.beta - 1.0)

        return g, Jg_J, Jg_JJ

    def get_principal_stretches(self, eps: float = 1e-6) -> torch.Tensor:
        C = self.kin.C
        trace_C = C.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace_C_squared = (C @ C).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        invariant_2 = 0.5 * (trace_C.square() - trace_C_squared)
        invariant_3 = torch.linalg.det(C)

        p = invariant_2 - trace_C.square() / 3.0
        q = -2.0 * trace_C.pow(3) / 27.0 + trace_C * invariant_2 / 3.0 - invariant_3

        mean = trace_C / 3.0
        scale = mean.abs().clamp_min(eps**0.5)
        near_triple = -p <= eps * scale.square()

        p_for_roots = torch.minimum(p, -eps * scale.square())
        radius = torch.sqrt(-p_for_roots / 3.0)
        cos_argument = -q / (2.0 * radius.pow(3))
        safe_cos_argument = cos_argument.clamp(min=-1.0 + eps, max=1.0 - eps)
        theta = torch.acos(safe_cos_argument) / 3.0
        offsets = torch.arange(3, dtype=C.dtype, device=C.device) * (
            2.0 * torch.pi / 3.0
        )
        eigenvalues = mean.unsqueeze(-1) + 2.0 * radius.unsqueeze(-1) * torch.cos(
            theta.unsqueeze(-1) - offsets
        )
        eigenvalues = eigenvalues.flip(dims=(-1,))

        near_double = cos_argument.abs() >= 1.0 - eps
        cube_root_arg = -0.5 * q
        cube_root = cube_root_arg.sign() * cube_root_arg.abs().clamp_min(
            torch.finfo(C.dtype).tiny
        ).pow(1.0 / 3.0)
        repeated = mean - cube_root
        unique = mean + 2.0 * cube_root
        double_eigenvalues = torch.stack(
            [
                torch.where(cube_root >= 0.0, repeated, unique),
                repeated,
                torch.where(cube_root >= 0.0, unique, repeated),
            ],
            dim=-1,
        )
        eigenvalues = torch.where(
            near_double.unsqueeze(-1), double_eigenvalues, eigenvalues
        )
        eigenvalues = torch.where(
            near_triple.unsqueeze(-1), mean.unsqueeze(-1), eigenvalues
        )
        return torch.sqrt(eigenvalues.clamp_min(eps))

    def get_psi(self) -> torch.Tensor:
        lam_princ = self.get_principal_stretches()

        lam_pow = lam_princ.unsqueeze(-1) ** self.alphas
        phi = (torch.sum(lam_pow, dim=-2) - 3.0) / self.alphas
        psi_deviatronic = torch.sum(
            self.mus * (phi - torch.log(self.kin.J).unsqueeze(-1)), dim=-1
        )
        psi_volumetric = self.lam * self.get_g()[0]

        self.psi = psi_deviatronic + psi_volumetric

        return self.psi

    def get_S(
        self,
    ) -> torch.Tensor:
        if self.psi is None:
            self.get_psi()

        S = torch.autograd.grad(
            outputs=self.psi.sum(),  # type: ignore
            inputs=self.kin.E,
            create_graph=True,
            retain_graph=True,
        )[0]
        return S

    def get_P(self) -> torch.Tensor:
        return torch.autograd.grad(
            outputs=self.psi.sum(),  # type: ignore
            inputs=self.kin.F,
            create_graph=True,
            retain_graph=True,
        )[0]

    def hills_constitutive_inequality(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, Jg_JJ = self.get_g()
        return (
            torch.all(self.mus * self.alphas > 0),
            torch.tensor(self.beta > 0, device=self.mus.device),
            torch.all(self.kin.J * Jg_JJ > 0),
        )
