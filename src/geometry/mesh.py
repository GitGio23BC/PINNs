from dataclasses import dataclass

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
    edges: torch.Tensor
    boundary_nodes: torch.Tensor

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]


def create_mesh(
    width: float = 1.0,
    height: float = 1.0,
    nx: int = 10,
    ny: int = 10,
    device: torch.device | str = "cpu",
):
    """
    Create a regular 2D triangular mesh.

    Parameters
    ----------
    width : float
        Width of the domain.

    height : float
        Height of the domain.

    nx : int
        Number of subdivisions along the x direction.

    ny : int
        Number of subdivisions along the y direction.

    Returns
    -------
    Mesh
        Generated triangular mesh.
    """
    x = torch.linspace(0.0, width, nx + 1, device=device, dtype=torch.float32)
    y = torch.linspace(0.0, height, ny + 1, device=device, dtype=torch.float32)

    Y, X = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    elements = []

    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1

            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])

    elements = torch.tensor(elements, dtype=torch.int64)

    all_edges = torch.cat([
        elements[:, [0, 1]],
        elements[:, [1, 2]],
        elements[:, [2, 0]]
    ], dim=0)
    all_edges, _ = torch.sort(all_edges, dim=1)
    edges = torch.unique(all_edges, dim=0)
    edeges = edges.to(device=device, dtype=torch.int64)

    tol = 1e-12
    boundary_mask = (
        (nodes[:, 0] <= tol) |
        (nodes[:, 0] >= width - tol) |
        (nodes[:, 1] <= tol) |
        (nodes[:, 1] >= height - tol)
    )
    boundary_nodes = torch.nonzero(boundary_mask, as_tuple=False).squeeze(1)

    return Mesh(
        nodes=nodes,
        elements=elements,
        edges=edges,
        boundary_nodes=boundary_nodes
    )
