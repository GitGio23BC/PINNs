import torch
from torch import nn

from .mesh_graph_net import MeshGraphNet, MeshGraphNetAn

FEATURE_DIMS = {
    "base": 6 + 2 + 1,
    "traction": 8 + 2 + 1,
    "dynamic": 10 + 2 + 1,
    "visco": 11 + 2 + 1,
    "full": 16 + 2 + 1,
}


def build_mgn_model(
    cfg: dict[str, dict],
    method,
    device: torch.device | str,
) -> nn.Module:
    m_cfg = cfg["model"]
    method = cfg["training"]["method"] if method is None else method
    node_in_dim = FEATURE_DIMS[method]

    edge_in_dim = int(m_cfg.get("edge_in_dim", 6))
    latent_dim = int(m_cfg.get("latent_dim", 128))
    hidden_dim = int(m_cfg.get("hidden_dim", 128))
    num_layers = int(m_cfg.get("num_layers", 15))
    output_dim = int(m_cfg.get("output_dim", 5))

    model_type = m_cfg["type"]

    if model_type == "standard":
        model_cls = MeshGraphNet
    elif model_type == "ansatz_space":
        model_cls = MeshGraphNetAn
    else:
        raise ValueError(
            f"Unknown MGN model type: '{model_type}'. Expected 'standard' or 'ansatz_space'."
        )

    return model_cls(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        activation=nn.SiLU,
    ).to(device)
