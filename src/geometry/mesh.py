from dataclasses import dataclass
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import torch
import numpy as np

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
    left_nodes: torch.Tensor
    right_nodes: torch.Tensor
    bottom_nodes: torch.Tensor
    top_nodes: torch.Tensor
    front_nodes: torch.Tensor
    back_nodes: torch.Tensor
    fixed_dofs: torch.Tensor
    free_dofs: torch.Tensor

    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0

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

    @property
    def n_dofs_3D(self) -> int:
        return 3 * self.nodes.shape[0]


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
    x = torch.linspace(0.0, width, nx + 1, device=device, dtype=torch.float64)
    y = torch.linspace(0.0, height, ny + 1, device=device, dtype=torch.float64)

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

    boundary_nodes = torch.nonzero(
        left_mask | right_mask | bottom_mask | top_mask, as_tuple=False
    ).squeeze(1)

    fixed_dofs = torch.cat(
        [2 * left_nodes, 2 * left_nodes + 1]
    ).sort().values

    total_dofs = 2 * nodes.shape[0]
    free_mask = torch.ones(total_dofs, dtype=torch.bool, device=device)
    free_mask[fixed_dofs] = False
    free_dofs = torch.nonzero(free_mask, as_tuple=False).squeeze(1)

    return Mesh(
        nodes=nodes,
        elements=elements,
        edges=edges,
        boundary_nodes=boundary_nodes,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        bottom_nodes=bottom_nodes,
        top_nodes=top_nodes,
        fixed_dofs=fixed_dofs,
        free_dofs=free_dofs,
    )

def create_mesh_3D(
    width: float = 20.0,
    height: float = 10.0,
    depth: float = 10.0,
    nx: int = 20,
    ny: int = 10,
    nz: int = 10,
    device: torch.device | str = "cpu",
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

    x = torch.linspace(0.0, width, nx+1, dtype=torch.float64)
    y = torch.linspace(0.0, height, ny+1, dtype=torch.float64)
    z = torch.linspace(0.0, depth, nz+1, dtype=torch.float64)

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    nodes_np = np.stack([X.reshape(-1).numpy(), Y.reshape(-1).numpy(), Z.reshape(-1).numpy()], axis=1)

    delaunay = Delaunay(nodes_np)
    elements_np = delaunay.simplices

    nodes = torch.tensor(nodes_np, dtype=torch.float64, device=device)
    elements = torch.tensor(elements_np, dtype=torch.int64, device=device)

    all_edges = torch.cat(
        [elements[:, [0, 1]], elements[:, [0, 2]], elements[:, [0, 3]],
         elements[:, [1, 2]], elements[:, [1, 3]], elements[:, [2, 3]]],
          dim=0
    )

    all_edges, _ = torch.sort(all_edges, dim=1)
    edges = torch.unique(all_edges, dim=0)
    edges = edges.to(device=device, dtype=torch.int64)

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

    boundary_nodes = torch.nonzero(
        left_mask | right_mask | bottom_mask | top_mask | front_mask | back_mask,
        as_tuple=False
    ).squeeze(1)

    fixed_dofs = torch.cat(
        [3 * left_nodes,
         3 * left_nodes + 1,
         3 * left_nodes + 2]
    ).sort().values

    total_dofs = 3 * nodes.shape[0]
    free_mask = torch.ones(total_dofs, dtype=torch.bool, device=device)
    free_mask[fixed_dofs] = False
    free_dofs = torch.nonzero(free_mask, as_tuple=False).squeeze(1)

    return Mesh(
        nodes=nodes,
        elements=elements,
        edges=edges,
        boundary_nodes=boundary_nodes,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        bottom_nodes=bottom_nodes,
        top_nodes=top_nodes,
        front_nodes=front_nodes,
        back_nodes=back_nodes,
        fixed_dofs=fixed_dofs,
        free_dofs=free_dofs,
        width=width,
        height=height,
        depth=depth
    )

def test_mesh_3D():
    """ 
    Esegue un test di generazione della mesh 3D e plotta 
    il risultato evidenziando il grafo completo e le facce.
    """
    print("Generazione della mesh 3D in corso...")
    # Creiamo una mesh a bassa risoluzione (3x3x3) per una visualizzazione chiara
    mesh = create_mesh_3D( 
        device="cpu"
    )
    
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
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title("Modello 3D Completo (World Edges statici)")
    
    # Plottiamo gli edge
    for edge in edges:
        n1, n2 = edge
        ax1.plot3D(
            [nodes[n1, 0], nodes[n2, 0]],
            [nodes[n1, 1], nodes[n2, 1]],
            [nodes[n1, 2], nodes[n2, 2]],
            color='gray', alpha=0.4, linewidth=0.5
        )
    # Plottiamo tutti i nodi
    ax1.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='black', s=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect((mesh.width, mesh.height, mesh.depth)) # (width, height, depth)
    # Funzione di supporto per plottare le facce opposte
    def plot_faces(ax, mask1, mask2, color1, color2, label1, label2):
        # Disegna tutti i nodi in trasparenza come riferimento
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='lightgray', s=10, alpha=0.1)
        
        # Disegna la faccia 1
        idx1 = mask1.cpu().numpy()
        ax.scatter(nodes[idx1, 0], nodes[idx1, 1], nodes[idx1, 2], c=color1, s=40, label=label1, depthshade=False)
        
        # Disegna la faccia 2
        idx2 = mask2.cpu().numpy()
        ax.scatter(nodes[idx2, 0], nodes[idx2, 1], nodes[idx2, 2], c=color2, s=40, label=label2, depthshade=False)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    # ==========================================
    # 2. Plot: Facce asse X (Left e Right)
    # ==========================================
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.set_title("Asse X: Facce Left (Ancoraggio) & Right (Libera)")
    plot_faces(ax2, mesh.left_nodes, mesh.right_nodes, 'red', 'blue', 'Left Nodes (Fissi)', 'Right Nodes')

    # ==========================================
    # 3. Plot: Facce asse Y (Bottom e Top)
    # ==========================================
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_title("Asse Y: Facce Bottom & Top")
    plot_faces(ax3, mesh.bottom_nodes, mesh.top_nodes, 'green', 'orange', 'Bottom Nodes', 'Top Nodes')

    # ==========================================
    # 4. Plot: Facce asse Z (Front e Back)
    # ==========================================
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title("Asse Z: Facce Front & Back")
    plot_faces(ax4, mesh.front_nodes, mesh.back_nodes, 'purple', 'cyan', 'Front Nodes', 'Back Nodes')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Assicurati di aver importato matplotlib.pyplot e mpl_toolkits.mplot3d in alto nel file
    test_mesh_3D()
