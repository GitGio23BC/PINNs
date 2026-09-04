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
    left_nodes: torch.Tensor
    right_nodes: torch.Tensor
    bottom_nodes: torch.Tensor
    top_nodes: torch.Tensor

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
    def boundary_nodes(self) -> torch.Tensor:
        return torch.unique(
            torch.cat(
                [
                    self.left_nodes,
                    self.right_nodes,
                    self.bottom_nodes,
                    self.top_nodes,
                ]
            )
        )


def create_mesh(
    width: float = 1.0,
    height: float = 1.0,
    nx: int = 10,
    ny: int = 10,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Mesh:

    x = torch.linspace(0, width, nx + 1, device=device, dtype=dtype)
    y = torch.linspace(0, height, ny + 1, device=device, dtype=dtype)

    Y, X = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((X.reshape(-1), Y.reshape(-1)), dim=1)

    index_dtype = torch.int64
    stride = nx + 1

    i = torch.arange(nx, device=device, dtype=index_dtype)
    j = torch.arange(ny, device=device, dtype=index_dtype)

    base = (j[:, None] * stride + i[None, :]).reshape(-1)

    n0 = base
    n1 = base + 1
    n2 = base + stride
    n3 = n2 + 1

    elements = torch.stack(
        (
            torch.stack((n0, n1, n3), dim=1),
            torch.stack((n0, n3, n2), dim=1),
        ),
        dim=1,
    ).reshape(-1, 3)

    row_starts = torch.arange(ny + 1, device=device, dtype=index_dtype) * stride
    col_starts = torch.arange(nx, device=device, dtype=index_dtype)

    horizontal_base = (row_starts[:, None] + col_starts[None, :]).reshape(-1)

    horizontal_edges = torch.stack((horizontal_base, horizontal_base + 1), dim=1)

    vertical_base = (
        torch.arange(ny, device=device, dtype=index_dtype)[:, None] * stride
        + torch.arange(nx + 1, device=device, dtype=index_dtype)[None, :]
    ).reshape(-1)

    vertical_edges = torch.stack((vertical_base, vertical_base + stride), dim=1)

    diagonal_edges = torch.stack((base, n3), dim=1)

    edges = torch.cat(
        (horizontal_edges, vertical_edges, diagonal_edges),
        dim=0,
    )

    row_indices = torch.arange(ny + 1, device=device, dtype=index_dtype)
    col_indices = torch.arange(nx + 1, device=device, dtype=index_dtype)

    left_nodes = row_indices * stride
    right_nodes = row_indices * stride + nx
    bottom_nodes = col_indices[1:]
    top_nodes = ny * stride + col_indices[1:]

    return Mesh(
        nodes=nodes,
        elements=elements,
        edges=edges,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        bottom_nodes=bottom_nodes,
        top_nodes=top_nodes,
    )
