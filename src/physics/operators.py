import torch

from src.geometry.mesh import Mesh


def compute_nodal_masses(mesh: Mesh, rho: float) -> torch.Tensor:
    nodal_masses = torch.zeros(
        mesh.n_nodes,
        dtype=mesh.nodes.dtype,
        device=mesh.nodes.device,
    )
    element_masses = rho * mesh.elements_area
    mass_per_corner = element_masses / 3.0

    for i in range(3):
        nodal_masses.scatter_add_(0, mesh.elements[:, i], mass_per_corner)

    return nodal_masses.unsqueeze(-1)


def compute_internal_nodal_forces(
    P: torch.Tensor,
    mesh: Mesh,
    dN_dX: torch.Tensor,
) -> torch.Tensor:
    f_int_local = torch.einsum("eij, eaj, e -> eai", P, dN_dX, mesh.elements_area)

    F_int = torch.zeros(
        (mesh.n_nodes, 2),
        dtype=P.dtype,
        device=P.device,
    )
    for i in range(3):
        F_int.scatter_add_(
            0,
            mesh.elements[:, i].unsqueeze(-1).expand(-1, 2),
            f_int_local[:, i, :],
        )

    return F_int


def pako_residual_2D(
    P: torch.Tensor,
    mesh: Mesh,
    dN_dX: torch.Tensor,
    b: torch.Tensor,
    rho: float,
    u_t: torch.Tensor,
    u_tt: torch.Tensor,
    alpha_rayleigh: float = 0.0,
    F_ext: torch.Tensor | None = None,
) -> torch.Tensor:
    F_int = compute_internal_nodal_forces(P, mesh, dN_dX)
    M_a = compute_nodal_masses(mesh, rho)  

    F_body = M_a * b.unsqueeze(0)
    F_inertia = M_a * u_tt
    
    F_damp = alpha_rayleigh * M_a * u_t

    R = F_int + F_inertia + F_damp - F_body

    if F_ext is not None:
        R = R - F_ext

    return R