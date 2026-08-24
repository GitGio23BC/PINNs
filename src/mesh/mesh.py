from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Mesh:
    """
    Represents a 2D triangular mesh.

    Attributes
    ----------
    nodes : np.ndarray
        Coordinates of mesh nodes.
        Shape: (N, 2)

    elements : np.ndarray
        Triangular elements.
        Each row contains the indices of the three nodes
        belonging to a triangle.
        Shape: (T, 3)

    edges : np.ndarray
        Unique edges connecting neighboring mesh nodes.
        Shape: (E, 2)

    boundary_nodes : np.ndarray
        Indices of nodes belonging to the boundary.
    """

    nodes: np.ndarray
    elements: np.ndarray
    edges: np.ndarray
    boundary_nodes: np.ndarray

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

    # ---------------------------------------------------------
    # 1. Generate mesh nodes
    # ---------------------------------------------------------

    x = np.linspace(0.0, width, nx + 1)
    y = np.linspace(0.0, height, ny + 1)

    X, Y = np.meshgrid(x, y)

    nodes = np.column_stack([X.ravel(), Y.ravel()])

    # ---------------------------------------------------------
    # 2. Generate triangular elements
    # ---------------------------------------------------------

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

    elements = np.array(elements, dtype=int)

    # ---------------------------------------------------------
    # 3. Extract unique mesh edges
    # ---------------------------------------------------------

    edges = set()

    for triangle in elements:
        a, b, c = triangle

        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    edges = np.array(sorted(edges), dtype=int)

    # ---------------------------------------------------------
    # 4. Identify boundary nodes
    # ---------------------------------------------------------

    boundary_nodes = []

    tolerance = 1e-12

    for index, (x_coord, y_coord) in enumerate(nodes):
        on_left = abs(x_coord - 0.0) < tolerance
        on_right = abs(x_coord - width) < tolerance
        on_bottom = abs(y_coord - 0.0) < tolerance
        on_top = abs(y_coord - height) < tolerance

        if on_left or on_right or on_bottom or on_top:
            boundary_nodes.append(index)

    boundary_nodes = np.array(boundary_nodes, dtype=int)

    # ---------------------------------------------------------
    # 5. Return Mesh object
    # ---------------------------------------------------------

    return Mesh(
        nodes=nodes, elements=elements, edges=edges, boundary_nodes=boundary_nodes
    )


def plot_mesh(mesh, show_nodes=True, show_node_indices=False):
    """
    Plot the triangular mesh.
    """

    _fig, ax = plt.subplots(figsize=(7, 7))

    # Draw triangular elements
    for triangle in mesh.elements:
        points = mesh.nodes[triangle]

        polygon = np.vstack([points, points[0]])

        ax.plot(polygon[:, 0], polygon[:, 1])

    # Draw nodes
    if show_nodes:
        ax.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], s=20)

    # Draw node indices
    if show_node_indices:
        for i, (x, y) in enumerate(mesh.nodes):
            ax.text(x, y, str(i), fontsize=8)

    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"2D Triangular Mesh | "
        f"Nodes: {mesh.n_nodes} | "
        f"Elements: {mesh.n_elements} | "
        f"Edges: {mesh.n_edges}"
    )

    ax.grid(True)

    plt.show()


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    mesh = create_mesh(width=1.0, height=1.0, nx=5, ny=5)

    print("===== MESH INFORMATION =====")

    print(f"Number of nodes: {mesh.n_nodes}")
    print(f"Number of elements: {mesh.n_elements}")
    print(f"Number of edges: {mesh.n_edges}")
    print(f"Number of boundary nodes: {len(mesh.boundary_nodes)}")

    print("\nFirst nodes:")
    print(mesh.nodes[:5])

    print("\nFirst elements:")
    print(mesh.elements[:5])

    print("\nFirst edges:")
    print(mesh.edges[:10])

    print("\nBoundary nodes:")
    print(mesh.boundary_nodes)

    plot_mesh(mesh, show_nodes=True, show_node_indices=True)
