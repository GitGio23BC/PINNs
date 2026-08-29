import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP used to decode latent node features into physical quantities."""

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


class Decoder(nn.Module):
    """
    MeshGraphNet decoder.

    Converts latent node features into nodal displacement increments.

    Input:
        node_features: (N, latent_dim)

    Output:
        displacement: (N, 2)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 2
    ):
        super().__init__()

        self.mlp = MLP(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_features)


if __name__ == "__main__":
    print("=" * 70)
    print("MESHGRAPHNET DECODER TEST")
    print("=" * 70)

    torch.manual_seed(42)

    num_nodes = 4
    latent_dim = 32
    hidden_dim = 64

    # Synthetic latent node state
    node_features = torch.randn(
        num_nodes,
        latent_dim
    )

    print("\nInput latent node features:")
    print(f"Shape: {node_features.shape}")

    # Create decoder
    decoder = Decoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=2
    )

    # Decode
    displacement = decoder(node_features)

    print("\nDecoded displacement u:")
    print(displacement)

    print("\nOutput shape:")
    print(displacement.shape)

    # Checks
    print("\n" + "-" * 70)
    print("CHECKS")
    print("-" * 70)

    shape_ok = (
        displacement.shape
        == (num_nodes, 2)
    )

    print(
        "Displacement shape: "
        + ("PASS" if shape_ok else "FAIL")
    )

    finite_ok = torch.isfinite(displacement).all()

    print(
        "Finite output values: "
        + ("PASS" if finite_ok else "FAIL")
    )

    nonzero_ok = not torch.allclose(
        displacement,
        torch.zeros_like(displacement)
    )

    print(
        "Non-zero displacement: "
        + ("PASS" if nonzero_ok else "FAIL")
    )

    # Verify that the decoder responds to different latent states
    modified_features = node_features.clone()
    modified_features[0] += 1.0

    modified_displacement = decoder(modified_features)

    response_ok = not torch.allclose(
        displacement,
        modified_displacement
    )

    print(
        "Decoder responds to latent state: "
        + ("PASS" if response_ok else "FAIL")
    )

    all_passed = (
        shape_ok
        and finite_ok
        and nonzero_ok
        and response_ok
    )

    print("\n" + "=" * 70)

    if all_passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")