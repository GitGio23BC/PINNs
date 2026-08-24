from abc import ABC, abstractmethod
import numpy as np


class ConstitutiveModel(ABC):
    """
    Abstract base class for hyperelastic constitutive models.

    Every constitutive model must provide:

        - strain_energy(F)
        - first_piola_kirchhoff(F)

    where F is the deformation gradient.
    """

    @abstractmethod
    def strain_energy(self, F):
        """
        Compute the strain energy density Ψ(F).
        """
        pass

    @abstractmethod
    def first_piola_kirchhoff(self, F):
        """
        Compute the first Piola-Kirchhoff stress P(F).
        """
        pass

    def cauchy_stress(self, F):
        """
        Compute the Cauchy stress from P.

        σ = (1 / J) P Fᵀ
        """

        P = self.first_piola_kirchhoff(F)

        J = np.linalg.det(F)

        if J <= 0:
            raise ValueError(
                "Invalid deformation gradient: det(F) must be positive."
            )

        return (P @ F.T) / J