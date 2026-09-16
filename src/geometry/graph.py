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
    u: torch.Tensor,
    node_type: torch.Tensor,
    u_dot: torch.Tensor | None = None,
    trac: torch.Tensor | None = None,
    E_prev: torch.Tensor | None = None,
    S_prev: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> Graph:

    feats = [u, node_type]

    if trac is not None:
        feats.append(trac)
    if u_dot is not None:
        feats.append(u_dot)
    if E_prev is not None:
        feats.append(E_prev)
    if S_prev is not None:
        feats.append(S_prev)
    if t is not None:
        t_tensor = t.view(1, 1).to(device=device, dtype=torch.float32)
        t_feat = t_tensor.expand(mesh.n_nodes, 1)
        feats.append(t_feat)

    ref = mesh.nodes.clone().detach()
    ref.requires_grad_(True)
    cur = ref + u

    feats.append(ref)
    node_features = torch.cat(feats, dim=-1).to(device=device, dtype=torch.float32)

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

    edge_features = torch.cat(
        [
            ref_rel,
            ref_dist,
            cur_rel,
            cur_dist,
        ],
        dim=-1,
    )
    edge_features = edge_features.to(device=device, dtype=torch.float32)

    return Graph(
        mesh_nodes=ref,
        senders=senders,
        receivers=receivers,
        node_features=node_features,
        edge_features=edge_features,
    )
