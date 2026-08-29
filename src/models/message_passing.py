import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron used by the
    edge and node update functions.
    """
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

    def forward(self, x):
        return self.network(x)
    
class MessagePassingBlock(nn.Module):
    """
    One message-passing block inspired by MeshGraphNet.
    The block performs:
        1. Edge update
        2. Message aggregation
        3. Node update
    Edge update:
        e'_ij = MLP_e(
            [v_i, v_j, e_ij]
        )

    Message aggregation:
        m_i = sum_j e'_ji

    Node update:
        v'_i = MLP_v(
            [v_i, m_i]
        )

    where:
        v_i = node features
        e_ij = edge features
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        # -----------------------------------------------------
        # Edge update network
        # -----------------------------------------------------
        self.edge_mlp = MLP(
            input_dim=2 * node_dim + edge_dim,
            hidden_dim=hidden_dim,
            output_dim=edge_dim
        )
        # -----------------------------------------------------
        # Node update network
        # ----------------------------------------------------
        self.node_mlp = MLP(
            input_dim=node_dim + edge_dim,
            hidden_dim=hidden_dim,
            output_dim=node_dim
        )
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor
    ):
        """
        Parameters
        ----------
        node_features : torch.Tensor
            Shape (N, node_dim)

        edge_features : torch.Tensor
            Shape (E, edge_dim)

        senders : torch.Tensor
            Shape (E,)

        receivers : torch.Tensor
            Shape (E,)
        Returns
        -------
        updated_nodes : torch.Tensor
            Shape (N, node_dim)
        updated_edges : torch.Tensor
            Shape (E, edge_dim)
        """
        # =====================================================
        # 1. EDGE UPDATE
        # =====================================================
        sender_features = node_features[senders]
        receiver_features = node_features[receivers]
        edge_input = torch.cat(
            [
                sender_features,
                receiver_features,
                edge_features
            ],
            dim=1
        )
        updated_edges = edge_features +  self.edge_mlp(edge_input)
        # =====================================================
        # 2. MESSAGE AGGREGATION
        # =====================================================
        num_nodes = node_features.shape[0]
        aggregated_messages = torch.zeros(
            (num_nodes, updated_edges.shape[1]),
            dtype=updated_edges.dtype,
            device=updated_edges.device
        )
        aggregated_messages.index_add_(
            0,
            receivers,
            updated_edges
        )
        # =====================================================
        # 3. NODE UPDATE
        # =====================================================
        node_input = torch.cat(
            [
                node_features,
                aggregated_messages
            ],
            dim=1
        )
        updated_nodes = node_features + self.node_mlp(node_input)
        return updated_nodes, updated_edges
# =============================================================
# TEST
# =============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MESSAGE PASSING MODULE TEST")
    print("=" * 70)
    # ---------------------------------------------------------
    # Synthetic graph
    # ---------------------------------------------------------
    # Four nodes
    num_nodes = 4
    # Six directed edges
    senders = torch.tensor(
        [0, 1, 1, 2, 2, 3],
        dtype=torch.long
    )
    receivers = torch.tensor(
        [1, 0, 2, 1, 3, 2],
        dtype=torch.long
    )
    num_edges = len(senders)
    # ---------------------------------------------------------
    # Feature dimensions
    # ---------------------------------------------------------
    node_dim = 6
    edge_dim = 6
    hidden_dim = 32
    # ---------------------------------------------------------
    # Random input features
    # ---------------------------------------------------------
    torch.manual_seed(42)
    node_features = torch.randn(
        num_nodes,
        node_dim
    )
    edge_features = torch.randn(
        num_edges,
        edge_dim
    )
    print("\nInput node features:")
    print(f"Shape: {node_features.shape}")
    print("\nInput edge features:")
    print(f"Shape: {edge_features.shape}")
    print("\nNumber of edges:")
    print(num_edges)
    # ---------------------------------------------------------
    # Create message-passing block
    # ---------------------------------------------------------
    model = MessagePassingBlock(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim
    )
    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    updated_nodes, updated_edges = model(
        node_features,
        edge_features,
        senders,
        receivers
    )
    print("\nUpdated node features:")
    print(f"Shape: {updated_nodes.shape}")
    print("\nUpdated edge features:")
    print(f"Shape: {updated_edges.shape}")
    # =========================================================
    # CHECKS
    # =========================================================
    print("\n" + "-" * 70)
    print("CHECKS")
    print("-" * 70)
    # ---------------------------------------------------------
    # Check node dimensions
    # ---------------------------------------------------------
    node_shape_ok = (
        updated_nodes.shape
        == node_features.shape
    )
    print(
        "Node feature shape: "
        + ("PASS" if node_shape_ok else "FAIL")
    )
    # ---------------------------------------------------------
    # Check edge dimensions
    # ---------------------------------------------------------
    edge_shape_ok = (
        updated_edges.shape
        == edge_features.shape
    )
    print(
        "Edge feature shape: "
        + ("PASS" if edge_shape_ok else "FAIL")
    )
    # ---------------------------------------------------------
    # Check finite values
    # ---------------------------------------------------------
    finite_ok = (
        torch.isfinite(updated_nodes).all()
        and torch.isfinite(updated_edges).all()
    )
    print(
        "Finite output values: "
        + ("PASS" if finite_ok else "FAIL")
    )
    # ---------------------------------------------------------
    # Check that the block actually transforms
    # the input features
    # ---------------------------------------------------------
    nodes_changed = not torch.allclose(
        updated_nodes,
        node_features
    )
    edges_changed = not torch.allclose(
        updated_edges,
        edge_features
    )
    print(
        "Node features updated: "
        + ("PASS" if nodes_changed else "FAIL")
    )
    print(
        "Edge features updated: "
        + ("PASS" if edges_changed else "FAIL")
    )
    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    all_passed = (
        node_shape_ok
        and edge_shape_ok
        and finite_ok
        and nodes_changed
        and edges_changed
    )
    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")