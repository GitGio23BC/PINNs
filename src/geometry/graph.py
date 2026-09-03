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


def create_time_graph(
    mesh: Mesh,
    u: torch.Tensor,
    t: torch.Tensor,
    device: torch.device | str = "cpu",
) -> Graph:

    ref = mesh.nodes.clone().detach()
    ref.requires_grad_(True)
    cur = ref + u

    num_nodes = ref.shape[0]
    t_feat = t.view(1, 1).expand(num_nodes, 1)

    node_features = torch.cat([ref, cur, u, t_feat], dim=-1)

    senders = []
    receivers = []

    senders = torch.cat([mesh.edges[:, 0], mesh.edges[:, 1]], dim=0).to(
        device=device, dtype=torch.long
    )
    receivers = torch.cat([mesh.edges[:, 1], mesh.edges[:, 0]], dim=0).to(
        device=device, dtype=torch.long
    )

    ref_rel = ref[senders] - ref[receivers]
    ref_dist = torch.linalg.vector_norm(ref_rel, dim=-1, keepdim=True)

    cur_rel = cur[senders] - cur[receivers]
    cur_dist = torch.linalg.vector_norm(cur_rel, dim=-1, keepdim=True)

    edge_features = torch.cat([ref_rel, ref_dist, cur_rel, cur_dist, ], dim=-1)
    edge_features = edge_features.to(device=device, dtype=torch.float32)

    return Graph(
        mesh_nodes=ref,
        senders=senders,
        receivers=receivers,
        node_features=node_features,
        edge_features=edge_features,
    )


def create_graph(
    mesh: Mesh,
    u: torch.Tensor,
    t: float | torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> Graph:

    ref = mesh.nodes.clone().detach()
    ref.requires_grad_(True)
    cur = ref + u


    node_features = torch.cat([ref, cur, u], dim=-1)

    senders = []
    receivers = []

    senders = torch.cat([mesh.edges[:, 0], mesh.edges[:, 1]], dim=0).to(
        device=device, dtype=torch.long
    )
    receivers = torch.cat([mesh.edges[:, 1], mesh.edges[:, 0]], dim=0).to(
        device=device, dtype=torch.long
    )

    ref_rel = ref[senders] - ref[receivers]
    ref_dist = torch.linalg.vector_norm(ref_rel, dim=-1, keepdim=True)

    cur_rel = cur[senders] - cur[receivers]
    cur_dist = torch.linalg.vector_norm(cur_rel, dim=-1, keepdim=True)

    edge_features = torch.cat([ref_rel, ref_dist, cur_rel, cur_dist], dim=-1)
    edge_features = edge_features.to(device=device, dtype=torch.float32)

    return Graph(
        mesh_nodes=ref,
        senders=senders,
        receivers=receivers,
        node_features=node_features,
        edge_features=edge_features,
    )
