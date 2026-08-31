from .architectures import ParametricPINN
from .backbones import BACKBONE_REGISTRY
from .mesh_graph_net import MeshGraphNet

__all__ = [
    "BACKBONE_REGISTRY",
    "MeshGraphNet",
    "ParametricPINN",
]
