import torch
import torch.nn as nn
from src.models.message_passing import MessagePassingBlock


class Processor(nn.Module):
    """
    Processor della pipeline MeshGraphNet.

    Il processor:

        1. codifica le feature dei nodi;
        2. codifica le feature degli archi;
        3. incorpora, quando disponibile, il campo di
           spostamento u proveniente dalla PINN;
        4. applica una sequenza di blocchi di message passing.

    La connettività del grafo è descritta da:

        senders
        receivers

    Il processor non calcola direttamente la risposta
    costitutiva. La parte meccanica rimane separata nei
    moduli di src.physics e src.mechanics.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_message_passing_steps: int = 3,
        displacement_dim: int = 2
    ):
        """
        Parameters
        ----------
        node_feature_dim : int
            Numero di feature associate a ciascun nodo.

        edge_feature_dim : int
            Numero di feature associate a ciascun arco.

        hidden_dim : int
            Dimensione dello spazio latente.

        num_message_passing_steps : int
            Numero di blocchi di message passing.

        displacement_dim : int
            Dimensione del campo di spostamento prodotto
            dalla PINN.

            Per il nostro problema 2D:
                displacement_dim = 2
        """

        super().__init__()

        if num_message_passing_steps < 1:
            raise ValueError(
                "num_message_passing_steps must be >= 1."
            )
        if displacement_dim < 1:
            raise ValueError(
                "displacement_dim must be >= 1."
            )
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = (
            num_message_passing_steps
        )
        self.displacement_dim = displacement_dim
        # =====================================================
        # NODE ENCODER
        # =====================================================
        self.node_encoder = nn.Sequential(
            nn.Linear(
                node_feature_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )
        # =====================================================
        # EDGE ENCODER
        # =====================================================
        self.edge_encoder = nn.Sequential(
            nn.Linear(
                edge_feature_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )
        # =====================================================
        # DISPLACEMENT ENCODER
        #
        # Maps the PINN output:
        #
        #       u = [u_x, u_y]
        #
        # into the latent node space.
        # =====================================================
        self.displacement_encoder = nn.Sequential(
            nn.Linear(
                displacement_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )
        # =====================================================
        # MESSAGE PASSING PROCESSOR
        # =====================================================
        self.message_passing_blocks = nn.ModuleList(
            [
                MessagePassingBlock(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    hidden_dim=hidden_dim
                )
                for _ in range(
                    num_message_passing_steps
                )
            ]
        )
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        displacement: torch.Tensor | None = None
    ):
        """
        Process the graph.

        Parameters
        ----------
        node_features : torch.Tensor
            Initial node features.

            Shape:
                (N, node_feature_dim)

        edge_features : torch.Tensor
            Initial edge features.

            Shape:
                (E, edge_feature_dim)

        senders : torch.Tensor
            Sender node indices.

            Shape:
                (E,)

        receivers : torch.Tensor
            Receiver node indices.

            Shape:
                (E,)

        displacement : torch.Tensor, optional
            Displacement field u produced by the PINN.

            Shape:
                (N, displacement_dim)

            For the present 2D problem:

                u = [u_x, u_y]

            If omitted, the processor operates only on
            the graph features.

        Returns
        -------
        updated_nodes : torch.Tensor
            Processed node latent states.

            Shape:
                (N, hidden_dim)

        updated_edges : torch.Tensor
            Processed edge latent states.

            Shape:
                (E, hidden_dim)
        """
        # =====================================================
        # INPUT VALIDATION
        # =====================================================
        if node_features.ndim != 2:
            raise ValueError(
                "node_features must have shape (N, F)."
            )
        if edge_features.ndim != 2:
            raise ValueError(
                "edge_features must have shape (E, F)."
            )
        if senders.ndim != 1:
            raise ValueError(
                "senders must have shape (E,)."
            )
        if receivers.ndim != 1:
            raise ValueError(
                "receivers must have shape (E,)."
            )
        if senders.shape != receivers.shape:
            raise ValueError(
                "senders and receivers must have "
                "the same shape."
            )
        if senders.shape[0] != edge_features.shape[0]:
            raise ValueError(
                "Number of sender/receiver indices must "
                "match the number of edges."
            )
        if displacement is not None:
            if displacement.ndim != 2:
                raise ValueError(
                    "displacement must have shape (N, D)."
                )
            if displacement.shape[0] != node_features.shape[0]:
                raise ValueError(
                    "displacement must contain one "
                    "value for each node."
                )
            if displacement.shape[1] != self.displacement_dim:
                raise ValueError(
                    "displacement has an incorrect "
                    "feature dimension."
                )
        # =====================================================
        # ENCODE NODE FEATURES
        # =====================================================
        nodes = self.node_encoder(
            node_features
        )
        # =====================================================
        # ENCODE EDGE FEATURES
        # =====================================================
        edges = self.edge_encoder(
            edge_features
        )
        # =====================================================
        # INCORPORATE PINN DISPLACEMENT
        # =====================================================
        if displacement is not None:
            displacement = displacement.to(
                device=nodes.device,
                dtype=nodes.dtype
            )
            displacement_latent = (
                self.displacement_encoder(
                    displacement
                )
            )
            nodes = nodes + displacement_latent
        # =====================================================
        # MESSAGE PASSING
        # =====================================================
        for block in self.message_passing_blocks:
            nodes, edges = block(
                nodes,
                edges,
                senders,
                receivers
            )
        return nodes, edges
# =============================================================
# TEST
# =============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MESHGRAPHNET PROCESSOR TEST")
    print("=" * 70)
    torch.manual_seed(42)
    # =========================================================
    # SYNTHETIC GRAPH
    # =========================================================
    num_nodes = 4
    senders = torch.tensor(
        [0, 1, 1, 2, 2, 3],
        dtype=torch.long
    )
    receivers = torch.tensor(
        [1, 0, 2, 1, 3, 2],
        dtype=torch.long
    )
    num_edges = len(senders)
    # =========================================================
    # FEATURE DIMENSIONS
    # =========================================================
    node_feature_dim = 6
    edge_feature_dim = 6
    hidden_dim = 32
    # =========================================================
    # RANDOM GRAPH FEATURES
    # =========================================================
    node_features = torch.randn(
        num_nodes,
        node_feature_dim
    )
    edge_features = torch.randn(
        num_edges,
        edge_feature_dim
    )
    # =========================================================
    # SYNTHETIC PINN DISPLACEMENT
    #
    # This represents:
    #
    #       u(X_i)
    #
    # returned by the PINN.
    # =========================================================

    displacement = torch.tensor(
        [
            [0.00, 0.00],
            [0.10, 0.00],
            [0.00, 0.05],
            [0.10, 0.05]
        ],
        dtype=torch.float32
    )
    print("\nInput node features:")
    print(
        f"Shape: {node_features.shape}"
    )
    print("\nInput edge features:")
    print(
        f"Shape: {edge_features.shape}"
    )
    print("\nNumber of edges:")
    print(num_edges)
    print("\nPINN displacement u:")
    print(displacement)
    # =========================================================
    # CREATE PROCESSOR
    # =========================================================
    processor = MeshGraphNetProcessor(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        num_message_passing_steps=3,
        displacement_dim=2
    )
    # =========================================================
    # FORWARD PASS
    # =========================================================
    updated_nodes, updated_edges = processor(
        node_features,
        edge_features,
        senders,
        receivers,
        displacement
    )
    print("\nProcessed node features:")
    print(
        f"Shape: {updated_nodes.shape}"
    )
    print("\nProcessed edge features:")
    print(
        f"Shape: {updated_edges.shape}"
    )
    # =========================================================
    # CHECKS
    # =========================================================
    print("\n" + "-" * 70)
    print("CHECKS")
    print("-" * 70)
    # ---------------------------------------------------------
    # Node output shape
    # ---------------------------------------------------------
    node_shape_ok = (
        updated_nodes.shape
        ==
        (num_nodes, hidden_dim)
    )
    print(
        "Node output shape: "
        +
        ("PASS" if node_shape_ok else "FAIL")
    )
    # ---------------------------------------------------------
    # Edge output shape
    # ---------------------------------------------------------
    edge_shape_ok = (
        updated_edges.shape
        ==
        (num_edges, hidden_dim)
    )
    print(
        "Edge output shape: "
        +
        ("PASS" if edge_shape_ok else "FAIL")
    )
    # ---------------------------------------------------------
    # Finite values
    # ---------------------------------------------------------
    finite_ok = (
        torch.isfinite(updated_nodes).all()
        and
        torch.isfinite(updated_edges).all()
    )
    print(
        "Finite output values: "
        +
        ("PASS" if finite_ok else "FAIL")
    )
    # ---------------------------------------------------------
    # Nodes changed
    # ---------------------------------------------------------
    nodes_changed = not torch.allclose(
        updated_nodes,
        torch.zeros_like(updated_nodes)
    )
    print(
        "Node processing: "
        +
        ("PASS" if nodes_changed else "FAIL")
    )
    # ---------------------------------------------------------
    # Edges changed
    # ---------------------------------------------------------
    edges_changed = not torch.allclose(
        updated_edges,
        torch.zeros_like(updated_edges)
    )
    print(
        "Edge processing: "
        +
        ("PASS" if edges_changed else "FAIL")
    )
    # ---------------------------------------------------------
    # PINN displacement actually affects the processor
    # ---------------------------------------------------------
    nodes_without_displacement, _ = processor(
        node_features,
        edge_features,
        senders,
        receivers,
        displacement=None
    )
    displacement_effect = not torch.allclose(
        updated_nodes,
        nodes_without_displacement
    )
    print(
        "PINN displacement affects state: "
        +
        (
            "PASS"
            if displacement_effect
            else "FAIL"
        )
    )
    # =========================================================
    # FINAL RESULT
    # =========================================================
    all_passed = (
        node_shape_ok
        and edge_shape_ok
        and finite_ok
        and nodes_changed
        and edges_changed
        and displacement_effect
    )
    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")