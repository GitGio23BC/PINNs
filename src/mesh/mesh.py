from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch


@dataclass
class Mesh:
    """
    Represents a 2D triangular mesh.

    Attributes
    ----------
    nodes : torch.Tensor
        Coordinates of mesh nodes.
        Shape: (N, 2)

    elements : torch.Tensor
        Triangular elements.
        Each row contains the indices of the three nodes
        belonging to a triangle.
        Shape: (T, 3)

    edges : torch.Tensor
        Unique edges connecting neighboring mesh nodes.
        Shape: (E, 2)

    boundary_nodes : torch.Tensor
        Indices of nodes belonging to the boundary.
    """

    nodes: torch.Tensor
    elements: torch.Tensor
    edges: torch.Tensor
    boundary_nodes: torch.Tensor

    @property
    def n_nodes(self):
        """Number of mesh nodes."""
        return len(self.nodes)

    @property
    def n_elements(self):
        """Number of triangular elements."""
        return len(self.elements)

    @property
    def n_edges(self):
        """Number of unique mesh edges."""
        return len(self.edges)


def create_mesh(width=1.0, height=1.0, nx=10, ny=10):
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
    x = torch.linspace(0.0, width, nx + 1)
    y = torch.linspace(0.0, height, ny + 1)

    X, Y = torch.meshgrid(x, y, indexing='ij')


    nodes = torch.column_stack([X.ravel(), Y.ravel()])

    elements = []

    for j in range(ny):
        for i in range(nx):
            # Node indices of the current rectangular cell
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1

            # Split the rectangle into two triangles
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])

    elements = torch.tensor(elements, dtype=torch.int64)

    edges = set()

    for triangle in elements:
        a, b, c = triangle

        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    edges = torch.tensor(sorted(edges), dtype=torch.int64)

    boundary_nodes = []

    tolerance = 1e-12

    for index, (x_coord, y_coord) in enumerate(nodes):
        on_left = abs(x_coord - 0.0) < tolerance
        on_right = abs(x_coord - width) < tolerance
        on_bottom = abs(y_coord - 0.0) < tolerance
        on_top = abs(y_coord - height) < tolerance

        if on_left or on_right or on_bottom or on_top:
            boundary_nodes.append(index)

    boundary_nodes = torch.tensor(boundary_nodes, dtype=torch.int64)

    return Mesh(
        nodes=nodes, elements=elements, edges=edges, boundary_nodes=boundary_nodes
    )