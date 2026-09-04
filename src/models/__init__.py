from .architectures import ParametricPINN
from .backbones import BACKBONE_REGISTRY

__all__ = [
    "BACKBONE_REGISTRY",
    "ParametricPINN",
]
