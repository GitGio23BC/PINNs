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
    elements_area: torch.Tensor
    edges: torch.Tensor
    boundary_nodes: torch.Tensor
    left_nodes: torch.Tensor
    right_nodes: torch.Tensor
    bottom_nodes: torch.Tensor
    top_nodes: torch.Tensor
    left_elements: torch.Tensor
    right_elements: torch.Tensor
    bottom_elements: torch.Tensor
    top_elements: torch.Tensor
    left_lengths: torch.Tensor
    right_lengths: torch.Tensor
    bottom_lengths: torch.Tensor
    top_lengths: torch.Tensor
    fixed_dofs: torch.Tensor
    free_dofs: torch.Tensor
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
        return 2 * self.nodes.shape[0]


def get_face_length(face_nodes, nodes):
    if len(face_nodes) < 2:
        return torch.zeros(len(face_nodes), dtype=nodes.dtype, device=nodes.device)
    coords = nodes[face_nodes]
    edge_lengths = torch.linalg.vector_norm(coords[1:] - coords[:-1], dim=1)

    lengths = torch.zeros(len(face_nodes), dtype=nodes.dtype, device=nodes.device)
    lengths[0] = 0.5 * edge_lengths[0]
    lengths[-1] = 0.5 * edge_lengths[-1]
    lengths[1:-1] = 0.5 * (edge_lengths[:-1] + edge_lengths[1:])
    return lengths


def create_mesh(
    width: float = 1.0,
    height: float = 1.0,
    nx: int = 10,
    ny: int = 10,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
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
    x = torch.linspace(0.0, width, nx + 1, device=device, dtype=dtype)
    y = torch.linspace(0.0, height, ny + 1, device=device, dtype=dtype)

    Y, X = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    elements = []

    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1

            # Alternate the diagonal because using a single direction produces a anisotropy
            if (i + j) % 2 == 0:
                elements.append([n0, n1, n3])
                elements.append([n0, n3, n2])
            else:
                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])

    elements = torch.tensor(elements, dtype=torch.int64, device=device)

    all_edges = torch.cat(
        [elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]], dim=0
    )
    all_edges, _ = torch.sort(all_edges, dim=1)
    edges = torch.unique(all_edges, dim=0)
    edges = edges.to(device=device, dtype=torch.int64)

    tol = 1e-12
    left_mask = nodes[:, 0] <= tol
    right_mask = nodes[:, 0] >= width - tol
    bottom_mask = nodes[:, 1] <= tol
    top_mask = nodes[:, 1] >= height - tol

    left_nodes = torch.nonzero(left_mask, as_tuple=False).squeeze(1)
    right_nodes = torch.nonzero(right_mask, as_tuple=False).squeeze(1)
    bottom_nodes = torch.nonzero(bottom_mask, as_tuple=False).squeeze(1)
    top_nodes = torch.nonzero(top_mask, as_tuple=False).squeeze(1)

    left_nodes = left_nodes[torch.argsort(nodes[left_nodes, 1])]
    right_nodes = right_nodes[torch.argsort(nodes[right_nodes, 1])]
    bottom_nodes = bottom_nodes[torch.argsort(nodes[bottom_nodes, 0])]
    top_nodes = top_nodes[torch.argsort(nodes[top_nodes, 0])]

    boundary_nodes = torch.nonzero(
        left_mask | right_mask | bottom_mask | top_mask, as_tuple=False
    ).squeeze(1)

    left_elements = torch.nonzero(
        left_mask[elements].sum(dim=1) == 2, as_tuple=False
    ).squeeze(1)
    right_elements = torch.nonzero(
        right_mask[elements].sum(dim=1) == 2, as_tuple=False
    ).squeeze(1)
    bottom_elements = torch.nonzero(
        bottom_mask[elements].sum(dim=1) == 2, as_tuple=False
    ).squeeze(1)
    top_elements = torch.nonzero(
        top_mask[elements].sum(dim=1) == 2, as_tuple=False
    ).squeeze(1)

    # Geometry terms
    ## Elements Areas = |(p1-p0) x (p2-po)|/2
    p0 = nodes[elements[:, 0]]
    p1 = nodes[elements[:, 1]]
    p2 = nodes[elements[:, 2]]

    Ae = 0.5 * (
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    elements_area = torch.abs(Ae)

    ## Border length
    right_lengths = get_face_length(right_nodes, nodes)
    left_lengths = get_face_length(left_nodes, nodes)
    bottom_lengths = get_face_length(bottom_nodes, nodes)
    top_lengths = get_face_length(top_nodes, nodes)

    ## Free and Fixed DOF (Diriclet)
    left_x_dofs = 2 * left_nodes

    mid_idx = len(left_nodes) // 2
    mid_left_node = left_nodes[mid_idx]
    mid_left_y_dof = 2 * mid_left_node + 1

    fixed_dofs = torch.cat([left_x_dofs, mid_left_y_dof.unsqueeze(0)]).sort().values

    total_dofs = 2 * nodes.shape[0]
    free_mask = torch.ones(total_dofs, dtype=torch.bool, device=device)
    free_mask[fixed_dofs] = False
    free_dofs = torch.nonzero(free_mask, as_tuple=False).squeeze(1)

    return Mesh(
        nodes=nodes,
        elements=elements,
        elements_area=elements_area,
        edges=edges,
        boundary_nodes=boundary_nodes,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        bottom_nodes=bottom_nodes,
        top_nodes=top_nodes,
        left_elements=left_elements,
        right_elements=right_elements,
        bottom_elements=bottom_elements,
        top_elements=top_elements,
        left_lengths=left_lengths,
        right_lengths=right_lengths,
        bottom_lengths=bottom_lengths,
        top_lengths=top_lengths,
        fixed_dofs=fixed_dofs,
        free_dofs=free_dofs,
    )
