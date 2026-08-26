import numpy as np
import torch

from .constitutive import ConstitutiveModel, STrainEnergy


# fmt: off
class HGO(ConstitutiveModel):
    """
    Simplified Holzapfel-Gasser-Ogden (HGO)
    hyperelastic model with two symmetric
    fiber families.

    The material consists of:

        - an isotropic Neo-Hookean matrix
        - two anisotropic fiber families

    The fiber directions are defined in the
    reference configuration.
    """

    def __init__(
        self,
        mu,
        bulk_modulus,
        k1,
        k2,
        fiber_angle
    ):
        """
        Parameters
        ----------
        mu : float
            Matrix shear modulus.

        bulk_modulus : float
            Bulk modulus.

        k1 : float
            Fiber stiffness parameter.

        k2 : float
            Fiber nonlinearity parameter.

        fiber_angle : float
            Fiber angle in radians.
        """

        self.mu = mu
        self.K = bulk_modulus

        self.k1 = k1
        self.k2 = k2

        self.fiber_angle = fiber_angle

        # -----------------------------------------------------
        # Reference fiber directions
        # -----------------------------------------------------

        alpha = fiber_angle

        self.a0 = np.array([
            np.cos(alpha),
            np.sin(alpha),
            0.0
        ])

        self.b0 = np.array([
            np.cos(alpha),
            -np.sin(alpha),
            0.0
        ])

    @staticmethod
    def macaulay(x):
        """
        Positive-part operator:

            <x> = max(x, 0)

        Fibers contribute only when stretched.
        """

        return max(x, 0.0)

    def strain_energy(self, F):
        """
        Compute the total HGO strain energy density.

        Ψ = Ψ_matrix + Ψ_fibers
        """

        J = np.linalg.det(F)

        if J <= 0:
            raise ValueError(
                "Invalid deformation gradient."
            )

        C = F.T @ F

        I1 = np.trace(C)

        logJ = np.log(J)

        # -----------------------------------------------------
        # Isotropic matrix
        # -----------------------------------------------------

        psi_matrix = (
            0.5 * self.mu *
            (
                I1
                - 3.0
                - 2.0 * logJ
            )
            +
            0.5 * self.K * logJ**2
        )

        # -----------------------------------------------------
        # Fiber invariants
        # -----------------------------------------------------

        I4_a = self.a0 @ C @ self.a0
        I4_b = self.b0 @ C @ self.b0

        # Fiber strains
        E_a = 0.5 * (I4_a - 1.0)
        E_b = 0.5 * (I4_b - 1.0)

        # Only tensile fiber strain contributes
        E_a_plus = self.macaulay(E_a)
        E_b_plus = self.macaulay(E_b)

        # -----------------------------------------------------
        # Fiber energies
        # -----------------------------------------------------

        psi_fiber_a = (
            self.k1 / (2.0 * self.k2)
            *
            (
                np.exp(
                    self.k2 * E_a_plus**2
                )
                - 1.0
            )
        )

        psi_fiber_b = (
            self.k1 / (2.0 * self.k2)
            *
            (
                np.exp(
                    self.k2 * E_b_plus**2
                )
                - 1.0
            )
        )

        psi_fibers = (
            psi_fiber_a
            +
            psi_fiber_b
        )

        return psi_matrix + psi_fibers

    def first_piola_kirchhoff(self, F):
        """
        Compute the first Piola-Kirchhoff stress.

        P = ∂Ψ / ∂F
        """

        J = np.linalg.det(F)

        if J <= 0:
            raise ValueError(
                "Invalid deformation gradient."
            )

        C = F.T @ F

        I1 = np.trace(C)  # noqa: F841

        F_inv_T = np.linalg.inv(F).T

        logJ = np.log(J)

        # -----------------------------------------------------
        # Matrix contribution
        # -----------------------------------------------------

        P_matrix = (
            self.mu *
            (
                F - F_inv_T
            )
            +
            self.K *
            logJ *
            F_inv_T
        )

        # -----------------------------------------------------
        # Fiber invariants
        # -----------------------------------------------------

        I4_a = self.a0 @ C @ self.a0
        I4_b = self.b0 @ C @ self.b0

        E_a = 0.5 * (I4_a - 1.0)
        E_b = 0.5 * (I4_b - 1.0)

        # -----------------------------------------------------
        # Fiber stresses
        # -----------------------------------------------------

        P_fiber_a = np.zeros((3, 3))
        P_fiber_b = np.zeros((3, 3))

        if E_a > 0:

            exponential_a = np.exp(
                self.k2 * E_a**2
            )

            coefficient_a = (
                self.k1
                * E_a
                * exponential_a
            )

            P_fiber_a = (
            coefficient_a
            *
            np.outer(
                F @ self.a0,
                self.a0
                )
            )

        if E_b > 0:

            exponential_b = np.exp(
                self.k2 * E_b**2
            )

            coefficient_b = (
                self.k1
                * E_b
                * exponential_b
            )

            P_fiber_b = (
                coefficient_b
                *
                np.outer(
                    F @ self.b0,
                    self.b0
                )
            )

        return (
            P_matrix
            +
            P_fiber_a
            +
            P_fiber_b
        )

# fmt: on


class FungEnergy_1D(STrainEnergy):
    def __init__(
        self,
        c: float,
        c1: float,
    ) -> None:

        self.c = c
        self.c1 = c1

    def energy(self, E: torch.Tensor):
        exp_E = torch.exp(self.c1 * E**2)
        return self.c * (exp_E - 1)

    def grad(self, E: torch.Tensor):
        exp_E = torch.exp(self.c1 * E**2)
        return 2.0 * self.c * self.c1 * E * exp_E

    def hessian(self, E: torch.Tensor):
        exp_E = torch.exp(self.c1 * E**2)
        return 2.0 * self.c * self.c1 * (1 - 2.0 * self.c1 * E**2) * exp_E


class FungEnergy_2D(STrainEnergy):
    def __init__(
        self,
        c: float,
        c1: float,
        c2: float,
        c3: float,
    ) -> None:

        self.c = c
        self.c1 = c1

    def energy(self, E: torch.Tensor) -> torch.Tensor:
        exp_E = torch.exp(self.c1 * E**2)
        return self.c * (exp_E - 1)

    def grad(self, E: torch.Tensor) -> torch.Tensor:
        exp_E = torch.exp(self.c1 * E**2)
        return 2.0 * self.c * self.c1 * E * exp_E

    def hessian(self, E: torch.Tensor) -> torch.Tensor:
        exp_E = torch.exp(self.c1 * E**2)
        return 2.0 * self.c * self.c1 * (1 - 2.0 * self.c1 * E**2) * exp_E


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

        grad_psi = self.psi.grad(E)

        H = self.psi.hessian(E)
        H_inv = torch.linalg.inv(H)
        H_inv2 = torch.bmm(H_inv, H_inv)

        return -self.k * H_inv2 * (grad_psi - S)
