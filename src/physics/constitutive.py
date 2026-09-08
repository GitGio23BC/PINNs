from abc import ABC, abstractmethod

import torch


class StrainEnergy(ABC):
    """
    Abstract base class for Strain Energy function.

    Every constitutive model must provide:

        - strain_energy(E)
        - strain_gradient(E)
        - strain_hessian(E)

    where F is the deformation gradient.
    """

    @abstractmethod
    def energy(self, E_voigt: torch.Tensor) -> torch.Tensor:
        """
        Strain Energy Function
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, E_voigt: torch.Tensor) -> torch.Tensor:
        """
        Strain Energy Gradient
        """
        raise NotImplementedError

    @abstractmethod
    def hessian(self, E_voigt: torch.Tensor) -> torch.Tensor:
        """
        Strain Energy Hessian
        """
        raise NotImplementedError
