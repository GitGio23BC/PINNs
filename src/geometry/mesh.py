from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch


@dataclass
class Mesh:
    """
    Represents a 2D triangular mesh.

    Attributes
    ----------
    nodes : np.narray
        Coordinates of mesh nodes.
        Shape: (N, 2)

    elements : np.narray
        Triangular elements.
        Each row contains the indices of the three nodes
        belonging to a triangle.
        Shape: (T, 3)

    edges : np.narray
        Unique edges connecting neighboring mesh nodes.
        Shape: (E, 2)

    boundary_nodes : np.narray
        Indices of nodes belonging to the boundary.
    """

    nodes: torch.Tensor
    elements: torch.Tensor
    elements_volume: torch.Tensor
    edges: torch.Tensor
    boundary_nodes: dict[str, torch.Tensor]
    boundary_elements: dict[str, torch.Tensor]
    boundary_faces: dict[str, torch.Tensor]
    boundary_areas: dict[str, torch.Tensor]
    fixed_dofs: torch.Tensor
    free_dofs: torch.Tensor

    width: float
    height: float
    depth: float

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def n_dofs(self) -> int:
        return 3 * self.nodes.shape[0]

    def get_boundary_nodes(
        self, direction: list[str] | str | None = None
    ) -> torch.Tensor:
        if direction is None:
            return torch.cat(list(self.boundary_nodes.values()), dim=0)
        if isinstance(direction, list):
            return torch.cat(
                [self.boundary_nodes[d] for d in direction],
                dim=0,
            )
        return self.boundary_nodes[direction]

    def get_boundary_elements(
        self, direction: list[str] | str | None = None
    ) -> torch.Tensor:
        if direction is None:
            return torch.cat(list(self.boundary_elements.values()), dim=0)
        if isinstance(direction, list):
            return torch.cat(
                [self.boundary_elements[d] for d in direction],
                dim=0,
            )
        return self.boundary_elements[direction]

    def get_boundary_faces(
        self, direction: list[str] | str | None = None
    ) -> torch.Tensor:
        if direction is None:
            return torch.cat(list(self.boundary_faces.values()), dim=0)
        if isinstance(direction, list):
            return torch.cat(
                [self.boundary_faces[d] for d in direction],
                dim=0,
            )
        return self.boundary_faces[direction]

    def get_boundary_areas(
        self, direction: list[str] | str | None = None
    ) -> torch.Tensor:
        if direction is None:
            return torch.cat(list(self.boundary_areas.values()), dim=0)
        if isinstance(direction, list):
            return torch.cat(
                [self.boundary_areas[d] for d in direction],
                dim=0,
            )
        return self.boundary_areas[direction]

    def get_boundary_nodal_areas(
        self, direction: list[str] | str | None = None
    ) -> torch.Tensor:
        right_faces = self.get_boundary_faces(direction)
        right_areas = self.get_boundary_areas(direction)

        nodal_areas = torch.zeros(
            self.n_nodes,
            dtype=right_areas.dtype,
            device=right_areas.device,
        )
        nodal_areas.index_add_(
            0,
            right_faces.reshape(-1),
            (right_areas / 3.0).repeat_interleave(3),
        )
        return nodal_areas


def create_mesh_3D(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 1.0,
    nx: int = 3,
    ny: int = 3,
    nz: int = 3,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Create a regular 3D tetrahedral mesh.

    Parameters
    ----------
    width : float
        Width of the domain.

    height : float
        Height of the domain.

    depth : float
        Depth of the domain.

    nx : int
        Number of subdivisions along the x direction.

    ny : int
        Number of subdivisions along the y direction.

    nz : int
        Number of subdivisions along the z direction.

    Returns
    -------
    Mesh
        Generated tetrahedral mesh.
    """

    x = torch.linspace(0.0, width, nx + 1, dtype=dtype, device=device)
    y = torch.linspace(0.0, height, ny + 1, dtype=dtype, device=device)
    z = torch.linspace(0.0, depth, nz + 1, dtype=dtype, device=device)

    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    nodes = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)

    grid_node_ids = torch.arange(
        (nx + 1) * (ny + 1) * (nz + 1), dtype=torch.long, device=device
    ).reshape(nx + 1, ny + 1, nz + 1)
    n000 = grid_node_ids[:-1, :-1, :-1]
    n100 = grid_node_ids[1:, :-1, :-1]
    n010 = grid_node_ids[:-1, 1:, :-1]
    n001 = grid_node_ids[:-1, :-1, 1:]
    n110 = grid_node_ids[1:, 1:, :-1]
    n101 = grid_node_ids[1:, :-1, 1:]
    n011 = grid_node_ids[:-1, 1:, 1:]
    n111 = grid_node_ids[1:, 1:, 1:]

    elements = torch.cat(
        [
            torch.stack(tetrahedron, dim=-1).reshape(-1, 4)
            for tetrahedron in (
                (n000, n100, n110, n111),
                (n000, n100, n101, n111),
                (n000, n001, n101, n111),
                (n000, n001, n011, n111),
                (n000, n010, n011, n111),
                (n000, n010, n110, n111),
            )
        ],
        dim=0,
    ).to(device=device)

    all_edges = torch.cat(
        [
            elements[:, [0, 1]],
            elements[:, [0, 2]],
            elements[:, [0, 3]],
            elements[:, [1, 2]],
            elements[:, [1, 3]],
            elements[:, [2, 3]],
        ],
        dim=0,
    ).to(device=device)

    all_edges, _ = torch.sort(all_edges, dim=1)
    edges = torch.unique(all_edges, dim=0)
    edges = edges.to(device=device)

    tol = 1e-12
    left_mask = nodes[:, 0] <= tol
    right_mask = nodes[:, 0] >= width - tol
    bottom_mask = nodes[:, 1] <= tol
    top_mask = nodes[:, 1] >= height - tol
    front_mask = nodes[:, 2] <= tol
    back_mask = nodes[:, 2] >= depth - tol

    left_nodes = torch.nonzero(left_mask, as_tuple=False).squeeze(1)
    right_nodes = torch.nonzero(right_mask, as_tuple=False).squeeze(1)
    bottom_nodes = torch.nonzero(bottom_mask, as_tuple=False).squeeze(1)
    top_nodes = torch.nonzero(top_mask, as_tuple=False).squeeze(1)
    front_nodes = torch.nonzero(front_mask, as_tuple=False).squeeze(1)
    back_nodes = torch.nonzero(back_mask, as_tuple=False).squeeze(1)

    nodes_by_direction = {
        "left": left_nodes,
        "right": right_nodes,
        "bottom": bottom_nodes,
        "top": top_nodes,
        "front": front_nodes,
        "back": back_nodes,
    }

    fixed_dofs = (
        torch.cat([3 * left_nodes, 3 * left_nodes + 1, 3 * left_nodes + 2])
        .sort()
        .values
    )

    total_dofs = 3 * nodes.shape[0]
    free_mask = torch.ones(total_dofs, dtype=torch.bool, device=device)
    free_mask[fixed_dofs] = False
    free_dofs = torch.nonzero(free_mask, as_tuple=False).squeeze(1)

    p0 = nodes[elements[:, 0]]
    p1 = nodes[elements[:, 1]]
    p2 = nodes[elements[:, 2]]
    p3 = nodes[elements[:, 3]]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0

    elements_volume = (
        torch.abs(torch.linalg.det(torch.stack([e1, e2, e3], dim=1))) / 6.0
    )

    face_ids = torch.arange(elements.shape[0], device=device, dtype=torch.long).repeat(
        4
    )

    element_faces = torch.cat(
        [
            elements[:, [0, 1, 2]],
            elements[:, [0, 1, 3]],
            elements[:, [0, 2, 3]],
            elements[:, [1, 2, 3]],
        ],
        dim=0,
    )
    element_faces = torch.sort(element_faces, dim=1).values

    all_faces, inverse, face_counts = torch.unique(
        element_faces, dim=0, return_inverse=True, return_counts=True
    )

    unique_face = torch.empty(all_faces.shape[0], device=device, dtype=torch.long)
    unique_face[inverse] = face_ids

    boundary_mask = face_counts == 1
    boundary_faces_by_nodes = all_faces[boundary_mask]
    boundary_faces_by_elements = unique_face[boundary_mask]

    points = nodes[boundary_faces_by_nodes]
    boundary_areas = (
        torch.linalg.vector_norm(
            torch.linalg.cross(
                points[:, 1] - points[:, 0], points[:, 2] - points[:, 0], dim=-1
            ),
            dim=-1,
        )
        / 2.0
    )

    boundary_faces_by_direction = {}
    boundary_areas_by_direction = {}
    boundary_face_elements_by_direction = {}

    for direction, direction_nodes in nodes_by_direction.items():
        on_direction = torch.isin(boundary_faces_by_nodes, direction_nodes).all(dim=1)
        boundary_faces_by_direction[direction] = boundary_faces_by_nodes[on_direction]
        boundary_areas_by_direction[direction] = boundary_areas[on_direction]
        boundary_face_elements_by_direction[direction] = boundary_faces_by_elements[
            on_direction
        ]

    return Mesh(
        nodes=nodes,
        elements=elements,
        elements_volume=elements_volume,
        edges=edges,
        boundary_nodes=nodes_by_direction,
        boundary_elements=boundary_face_elements_by_direction,
        boundary_faces=boundary_faces_by_direction,
        boundary_areas=boundary_areas_by_direction,
        fixed_dofs=fixed_dofs,
        free_dofs=free_dofs,
        width=width,
        height=height,
        depth=depth,
    )


def test_mesh_3D():
    """
    Esegue un test di generazione della mesh 3D e plotta
    il risultato evidenziando il grafo completo e le facce.
    """
    print("Generazione della mesh 3D in corso...")
    # Creiamo una mesh a bassa risoluzione (3x3x3) per una visualizzazione chiara
    mesh = create_mesh_3D(device="cpu")

    print(f"Nodi totali: {mesh.n_nodes}")
    print(f"Elementi (Tetraedri): {mesh.n_elements}")
    print(f"Edge totali: {mesh.n_edges}")
    print(f"Gradi di libertà (DOF): {mesh.n_dofs}")

    # Estraiamo i dati portandoli su CPU in formato NumPy per Matplotlib
    nodes = mesh.nodes.cpu().numpy()
    edges = mesh.edges.cpu().numpy()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("PIGNN - Test Inizializzazione Mesh 3D", fontsize=16)

    # ==========================================
    # 1. Plot: Modello 3D Completo (Nodi ed Edge)
    # ==========================================
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.set_title("Modello 3D Completo (World Edges statici)")

    # Plottiamo gli edge
    for edge in edges:
        n1, n2 = edge
        ax1.plot3D(
            [nodes[n1, 0], nodes[n2, 0]],
            [nodes[n1, 1], nodes[n2, 1]],
            [nodes[n1, 2], nodes[n2, 2]],
            color="gray",
            alpha=0.4,
            linewidth=0.5,
        )
    # Plottiamo tutti i nodi
    ax1.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="black", s=10)  # type: ignore
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_box_aspect((mesh.width, mesh.height, mesh.depth))  # (width, height, depth)

    # Funzione di supporto per plottare le facce opposte
    def plot_faces(ax, mask1, mask2, color1, color2, label1, label2):
        # Disegna tutti i nodi in trasparenza come riferimento
        ax.scatter(
            nodes[:, 0], nodes[:, 1], nodes[:, 2], c="lightgray", s=10, alpha=0.1
        )

        # Disegna la faccia 1
        idx1 = mask1.cpu().numpy()
        ax.scatter(
            nodes[idx1, 0],
            nodes[idx1, 1],
            nodes[idx1, 2],
            c=color1,
            s=40,
            label=label1,
            depthshade=False,
        )

        # Disegna la faccia 2
        idx2 = mask2.cpu().numpy()
        ax.scatter(
            nodes[idx2, 0],
            nodes[idx2, 1],
            nodes[idx2, 2],
            c=color2,
            s=40,
            label=label2,
            depthshade=False,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

    # ==========================================
    # 2. Plot: Facce asse X (Left e Right)
    # ==========================================
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.set_title("Asse X: Facce Left (Ancoraggio) & Right (Libera)")
    plot_faces(
        ax2,
        mesh.boundary_nodes["left"],
        mesh.boundary_nodes["right"],
        "red",
        "blue",
        "Left Nodes (Fissi)",
        "Right Nodes",
    )

    # ==========================================
    # 3. Plot: Facce asse Y (Bottom e Top)
    # ==========================================
    ax3 = fig.add_subplot(223, projection="3d")
    ax3.set_title("Asse Y: Facce Bottom & Top")
    plot_faces(
        ax3,
        mesh.boundary_nodes["bottom"],
        mesh.boundary_nodes["top"],
        "green",
        "orange",
        "Bottom Nodes",
        "Top Nodes",
    )

    # ==========================================
    # 4. Plot: Facce asse Z (Front e Back)
    # ==========================================
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.set_title("Asse Z: Facce Front & Back")
    plot_faces(
        ax4,
        mesh.boundary_nodes["front"],
        mesh.boundary_nodes["back"],
        "purple",
        "cyan",
        "Front Nodes",
        "Back Nodes",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_mesh_3D()
