import torch
from dataclasses import dataclass

@dataclass
class GraphFeatures:
    """
    Geometric and mechanical features associated
    with the nodes and edges of the graph.

    Node features:
        [X, Y, x, y, ux, uy]

    Edge features:
        [dX, dY, L0, dx, dy, L]

    where:
        X  = reference position
        x  = current position
        u  = displacement
        dX = reference relative position
        L0 = reference edge length
        dx = current relative position
        L  = current edge length
    """

    node_features: torch.Tensor
    edge_features: torch.Tensor

    @property
    def n_nodes(self):
        return self.node_features.shape[0]

    @property
    def n_node_features(self):
        return self.node_features.shape[1]

    @property
    def n_edges(self):
        return self.edge_features.shape[0]

    @property
    def n_edge_features(self):
        return self.edge_features.shape[1]

def create_features(
    graph,
    reference_nodes=None,
    current_nodes=None
):
    graph_nodes = torch.tensor(
        graph.nodes,
        dtype=torch.float32
    )

    if graph_nodes.ndim != 2 or graph_nodes.shape[1] != 2:
        raise ValueError(
            "graph.nodes must have shape (N, 2)."
        )

    N = graph_nodes.shape[0]

    if reference_nodes is None:
        reference_nodes = graph_nodes.copy()
    else:
        reference_nodes = torch.tensor(
            reference_nodes,
            dtype=torch.float32
        )

    if reference_nodes.shape != (N, 2):
        raise ValueError(
            "reference_nodes must have shape (N, 2)."
        )

    if current_nodes is None:
        current_nodes = graph_nodes.copy()
    else:
        current_nodes = torch.tensor(
            current_nodes,
            dtype=torch.float32
        )

    if current_nodes.shape != (N, 2):
        raise ValueError(
            "current_nodes must have shape (N, 2)."
        )

    displacement = (
        current_nodes
        -
        reference_nodes
    )

    node_features = torch.cat(
        [
            torch.tensor(reference_nodes, dtype=torch.float32),
            torch.tensor(current_nodes, dtype=torch.float32),
            torch.tensor(displacement, dtype=torch.float32)
        ],
        dim=1
    )

    senders = torch.tensor(
        graph.senders,
        dtype=torch.long
    )

    receivers = torch.tensor(
        graph.receivers,
        dtype=torch.long
    )

    if senders.shape != receivers.shape:
        raise ValueError(
            "graph.senders and graph.receivers "
            "must have the same shape."
        )

    reference_sender_positions = (
        reference_nodes[senders]
    )

    reference_receiver_positions = (
        reference_nodes[receivers]
    )

    reference_relative_position = (
        reference_receiver_positions
        -
        reference_sender_positions
    )

    reference_distance = torch.norm(
        reference_relative_position,
        p=2,
        dim=1,
        keepdim =True
    )

    current_sender_positions = (
        current_nodes[senders]
    )

    current_receiver_positions = (
        current_nodes[receivers]
    )

    current_relative_position = (
        current_receiver_positions
        -
        current_sender_positions
    )

    current_distance = torch.norm(
        current_relative_position,
        p=2,
        dim=1,
        keepdim =True
    )

    edge_features = torch.cat(
        [
            reference_relative_position,
            reference_distance,
            current_relative_position,
            current_distance
        ],
        dim=1
    )

    return GraphFeatures(
        node_features=node_features,
        edge_features=edge_features
    )