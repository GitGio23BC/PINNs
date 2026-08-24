import numpy as np

from .constitutive import ConstitutiveModel


class NeoHookean(ConstitutiveModel):
    """
    Compressible Neo-Hookean hyperelastic material.
    """

    def __init__(self, mu, bulk_modulus):
        self.mu = mu
        self.K = bulk_modulus

    def strain_energy(self, F):

        J = np.linalg.det(F)

        if J <= 0:
            raise ValueError(
                "Invalid deformation gradient: det(F) must be positive."
            )

        C = F.T @ F

        I1 = np.trace(C)

        return (
            0.5 * self.mu * (I1 - 3.0 - 2.0 * np.log(J))
            + 0.5 * self.K * (np.log(J)) ** 2
        )

    def first_piola_kirchhoff(self, F):

        J = np.linalg.det(F)

        if J <= 0:
            raise ValueError(
                "Invalid deformation gradient."
            )

        F_inv_T = np.linalg.inv(F).T

        return (
            self.mu * (F - F_inv_T)
            + self.K * np.log(J) * F_inv_T
        )