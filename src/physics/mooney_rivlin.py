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