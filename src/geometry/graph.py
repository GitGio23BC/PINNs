from dataclasses import dataclass

import torch

from .mesh import Mesh


@dataclass
class Graph:
    """
    Graph representation of a mesh.

    The graph is defined as:

        G = (V, E)

    where:

        V = mesh nodes
        E = directed mesh edges

    Attributes
    ----------
    nodes : np.ndarray
        Node coordinates.
        Shape: (N, 2)

    mesh_edges : np.ndarray
        Original undirected mesh edges.
        Shape: (E, 2)

    senders : np.ndarray
        Sender node for each directed edge.
        Shape: (2E,)

    receivers : np.ndarray
        Receiver node for each directed edge.
        Shape: (2E,)
    """

    mesh_nodes: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor

    @property
    def n_senders(self) -> int:
        return self.senders.shape[0]

    @property
    def n_receivers(self) -> int:
        return self.receivers.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.mesh_nodes.shape[0]

    @property
    def n_edges(self) -> int:
        return self.senders.shape[0]


def create_graph(
    mesh: Mesh,
    node_type: torch.Tensor,
    world_pos: torch.Tensor,
    device: torch.device | str = "cpu",
) -> Graph:

    u = mesh.nodes.clone().to(device=device, dtype=torch.float64)
    x = world_pos.clone().to(device=device, dtype=torch.float64)

    senders = torch.cat([mesh.edges[:, 0], mesh.edges[:, 1]], dim=0).to(
        device=device, dtype=torch.long
    )
    receivers = torch.cat([mesh.edges[:, 1], mesh.edges[:, 0]], dim=0).to(
        device=device, dtype=torch.long
    )

    u_rel = u[senders] - u[receivers]
    u_dist = torch.linalg.vector_norm(u_rel, dim=-1, keepdim=True)

    x_rel = x[senders] - x[receivers]
    x_dist = torch.linalg.vector_norm(x_rel, dim=-1, keepdim=True)

    node_type = node_type.to(device=device, dtype=torch.float64)
    node_features = torch.cat([node_type], dim=-1)

    edge_features = torch.cat([u_rel, u_dist, x_rel, x_dist], dim=-1)
    edge_features = edge_features.to(device=device, dtype=torch.float64)

    return Graph(
        mesh_nodes=u,
        senders=senders,
        receivers=receivers,
        node_features=node_features,
        edge_features=edge_features,
    )

