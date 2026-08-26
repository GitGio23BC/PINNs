from abc import ABC, abstractmethod

import numpy as np
import torch


# fmt: off
# ruff: disable[PIE790]
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
# fmt: on
# ruff: enable[PIE790]


class STrainEnergy(ABC):
    """
    Abstract base class for Strain Energy function.
    It is important to notice that the s- is privative
    so it is represent a Energy that if applied
    can remove the Train. From that STrainEnergy.

    Every constitutive model must provide:

        - strain_energy(E)
        - strain_gradient(E)
        - strain_hessian(E)

    where F is the deformation gradient.
    """

    @abstractmethod
    def energy(self, E: torch.Tensor) -> torch.Tensor:
        """
        Strain Energy Function
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, E: torch.Tensor) -> torch.Tensor:
        """
        Strain Energy Gradient
        """
        raise NotImplementedError

    @abstractmethod
    def hessian(self, E: torch.Tensor) -> torch.Tensor:
        """
        Strain Energy Hession
        """
        raise NotImplementedError


class 