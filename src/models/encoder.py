import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP used to encode node and edge features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Encoder(nn.Module):
    """
    MeshGraphNet encoder.

    Maps physical node and edge features into a common latent space.

    Input:
        node_features: (N, node_input_dim)
        edge_features: (E, edge_input_dim)

    Output:
        latent_nodes: (N, latent_dim)
        latent_edges: (E, latent_dim)
    """

    def __init__(
        self,
        node_input_dim: int = 6,
        edge_input_dim: int = 6,
        latent_dim: int = 32,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.node_encoder = MLP(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim
        )

        self.edge_encoder = MLP(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor
    ):
        latent_nodes = self.node_encoder(node_features)
        latent_edges = self.edge_encoder(edge_features)

        return latent_nodes, latent_edges


if __name__ == "__main__":
    print("=" * 70)
    print("MESHGRAPHNET ENCODER TEST")
    print("=" * 70)

    torch.manual_seed(42)

    num_nodes = 4
    num_edges = 6
    node_input_dim = 6
    edge_input_dim = 6
    latent_dim = 32
    hidden_dim = 64

    node_features = torch.randn(
        num_nodes,
        node_input_dim
    )

    edge_features = torch.randn(
        num_edges,
        edge_input_dim
    )

    print("\nInput node features:")
    print(f"Shape: {node_features.shape}")

    print("\nInput edge features:")
    print(f"Shape: {edge_features.shape}")

    encoder = Encoder(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    )

    latent_nodes, latent_edges = encoder(
        node_features,
        edge_features
    )

    print("\nEncoded node features:")
    print(f"Shape: {latent_nodes.shape}")

    print("\nEncoded edge features:")
    print(f"Shape: {latent_edges.shape}")

    print("\n" + "-" * 70)
    print("CHECKS")
    print("-" * 70)

    node_shape_ok = (
        latent_nodes.shape
        == (num_nodes, latent_dim)
    )

    print(
        "Node latent shape: "
        + ("PASS" if node_shape_ok else "FAIL")
    )

    edge_shape_ok = (
        latent_edges.shape
        == (num_edges, latent_dim)
    )

    print(
        "Edge latent shape: "
        + ("PASS" if edge_shape_ok else "FAIL")
    )

    finite_ok = (
        torch.isfinite(latent_nodes).all()
        and torch.isfinite(latent_edges).all()
    )

    print(
        "Finite output values: "
        + ("PASS" if finite_ok else "FAIL")
    )

    node_nonzero = not torch.allclose(
    latent_nodes,
    torch.zeros_like(latent_nodes)
    )

    edge_nonzero = not torch.allclose(
        latent_edges,
        torch.zeros_like(latent_edges)
    )

    print(
        "Node encoding non-zero: "
        + ("PASS" if node_nonzero else "FAIL")
    )

    print(
        "Edge encoding non-zero: "
        + ("PASS" if edge_nonzero else "FAIL")
    )

    modified_nodes = node_features.clone()
    modified_nodes[0] += 1.0

    modified_edges = edge_features.clone()
    modified_edges[0] += 1.0

    modified_latent_nodes, modified_latent_edges = encoder(
        modified_nodes,
        modified_edges
    )

    node_response = not torch.allclose(
        latent_nodes,
        modified_latent_nodes
    )

    edge_response = not torch.allclose(
        latent_edges,
        modified_latent_edges
    )

    print(
        "Node encoding responds to input: "
        + ("PASS" if node_response else "FAIL")
    )

    print(
        "Edge encoding responds to input: "
        + ("PASS" if edge_response else "FAIL")
    )

    all_passed = (
        node_shape_ok
        and edge_shape_ok
        and finite_ok
        and node_nonzero
        and edge_nonzero
        and node_response
        and edge_response
    )

    print("\n" + "=" * 70)

    if all_passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")


#   all_passed = (
#   node_shape_ok
#   and edge_shape_ok
#   and finite_ok
#   and node_changed
#   and edge_changed
#   )