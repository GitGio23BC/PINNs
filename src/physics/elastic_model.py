import numpy as np

from .constitutive import ConstitutiveModel


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

import numpy as np

from .constitutive import ConstitutiveModel


class MooneyRivlin(ConstitutiveModel):
    """
    Compressible two-parameter Mooney-Rivlin
    hyperelastic material.

    The formulation is normalized so that the
    reference configuration F = I is stress-free.
    """

    def __init__(self, C1, C2, bulk_modulus):

        self.C1 = C1
        self.C2 = C2
        self.K = bulk_modulus

    def strain_energy(self, F):
        """
        Compute the strain energy density Ψ(F).
        """

        J = np.linalg.det(F)

        if J <= 0:
            raise ValueError(
                "Invalid deformation gradient."
            )

        C = F.T @ F

        I1 = np.trace(C)

        I2 = 0.5 * (
            I1**2 -
            np.trace(C @ C)
        )

        logJ = np.log(J)

        return (
            self.C1 * (
                I1 - 3.0 - 2.0 * logJ
            )
            +
            self.C2 * (
                I2 - 3.0 - 4.0 * logJ
            )
            +
            0.5 * self.K * logJ**2
        )

    def first_piola_kirchhoff(self, F):
        """
        Compute the first Piola-Kirchhoff stress P.
        """

        J = np.linalg.det(F)

        if J <= 0:
            raise ValueError(
                "Invalid deformation gradient."
            )

        C = F.T @ F

        I1 = np.trace(C)

        F_inv_T = np.linalg.inv(F).T

        logJ = np.log(J)

        dI2_dF = (
            2.0 * (
                I1 * F
                -
                F @ C
            )
        )

        return (
            2.0 * self.C1 * (
                F - F_inv_T
            )
            +
            self.C2 * (
                dI2_dF
                -
                4.0 * F_inv_T
            )
            +
            self.K * logJ * F_inv_T
        )

import torch
import torch.nn as nn

class Viscoelastic(nn.Module):
    """
    Calcola il residuo dell'Equazione 33 (Haslach et al.) usando una funzione 
    di energia Psi calcolata esternamente (es. Fung, HGO, Neo-Hooke, o Neural Network).
    """

    def __init__(self, k_rate: float):
        super().__init__()
        # k_rate: costante cinetica k dell'Equazione 33
        self.k_rate = torch.tensor(k_rate, dtype=torch.float32)

    def forward(self, E: torch.Tensor, E_dot: torch.Tensor, S_ext: torch.Tensor, Psi: torch.Tensor) -> torch.Tensor:
        """
        Input:
            E: Tensore di deformazione di Green-Lagrange corrente [3, 3] (requires_grad=True)
            E_dot: Tensore dE/dt corrente [3, 3]
            S_ext: Tensore degli sforzi esterni applicati S(t) [3, 3]
            Psi: Valore scalare dell'energia di deformazione calcolato ESTERNAMENTE
            
        Output:
            residual: Tensore del residuo dell'Eq. 33 [3, 3]. Se il sistema rispetta la dinamica, vale 0.
        """
        # 1. Calcolo della prima derivata dPsi/dE (tensione interna di richiamo)
        dPsi_dE = torch.autograd.grad(Psi, E, create_graph=True)[0]

        # 2. Calcolo della seconda derivata d2Psi/dE2 (Hessiano [9, 9])
        dPsi_dE_flat = dPsi_dE.reshape(-1)
        num_el = dPsi_dE_flat.numel()
        hessian = torch.zeros(num_el, num_el, device=E.device, dtype=E.dtype)

        for i in range(num_el):
            grad_i = torch.autograd.grad(dPsi_dE_flat[i], E, retain_graph=True)[0]
            hessian[i] = grad_i.reshape(-1)

        # 3. Termine inversione dell'Hessiano al quadrato: H^(-1) @ H^(-1)
        H_inv = torch.linalg.pinv(hessian)
        H_inv_squared = H_inv @ H_inv

        # 4. Forzante fuori equilibrio: [-S(t) + dPsi/dE]
        force_term = -S_ext + dPsi_dE
        force_term_flat = force_term.reshape(-1, 1)

        # 5. Termine destro dell'Equazione 33: RHS = -k * (H^-2) * [-S(t) + dPsi/dE]
        rhs_flat = -self.k_rate * (H_inv_squared @ force_term_flat)
        rhs = rhs_flat.reshape(3, 3)

        # 6. RESIDUO: dE/dt - RHS = 0
        residual = E_dot - rhs

        return residual