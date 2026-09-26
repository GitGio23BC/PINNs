from pathlib import Path

import torch

from src.geometry.mesh import Mesh

from ..geometry import create_mesh
from ..physics import Kinematics, Ogden


def triangle_shape_gradients(
    nodes: torch.Tensor, elements: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    elem_coords = nodes[elements]
    x = elem_coords[:, :, 0]
    y = elem_coords[:, :, 1]

    # Compute the area using the determinant
    two_area = (
        x[:, 0] * (y[:, 1] - y[:, 2])
        + x[:, 1] * (y[:, 2] - y[:, 0])
        + x[:, 2] * (y[:, 0] - y[:, 1])
    )

    areas = 0.5 * two_area

    # Shape function derivatives dN_I / dX
    b = torch.stack(
        [
            y[:, 1] - y[:, 2],
            y[:, 2] - y[:, 0],
            y[:, 0] - y[:, 1],
        ],
        dim=1,
    ) / two_area.unsqueeze(1)

    # Shape function derivatives dN_I / dY
    c = torch.stack(
        [
            x[:, 2] - x[:, 1],
            x[:, 0] - x[:, 2],
            x[:, 1] - x[:, 0],
        ],
        dim=1,
    ) / two_area.unsqueeze(1)

    dN_dX = torch.stack([b, c], dim=-1)

    return dN_dX, areas


def compute_u_grad(
    u: torch.Tensor, elements: torch.Tensor, dN_dX: torch.Tensor
) -> torch.Tensor:

    # Get node displacement
    u_elem = u[elements]

    # grad_u = u_elem^T @ dN_dX for each node a
    grad_u = torch.einsum("eai, eaj -> eij", u_elem, dN_dX)

    return grad_u


def compute_external_force(
    t_n: torch.Tensor,
    T: float,
    mesh: Mesh,
    sigma_max: float,
    height: float,
    ny: int,
) -> torch.Tensor:
    F_ext = torch.zeros(mesh.n_dofs, dtype=torch.float64, device=mesh.nodes.device)

    traction_mag = sigma_max * torch.sin(
        torch.pi * torch.tensor([t_n]) / T
    )  # Magnitude (N/m)

    # Edge dimension i.e 1D area
    dy = height / ny

    # Sorting by y-coordinate
    right_node_indices = mesh.right_nodes
    y_coords = mesh.nodes[right_node_indices, 1]
    sorted_order = torch.argsort(y_coords)
    sorted_right_nodes = right_node_indices[sorted_order]

    num_r_nodes = len(sorted_right_nodes)

    for idx, node_id in enumerate(sorted_right_nodes):
        segment_length = 0.5 * dy if idx in (0, num_r_nodes - 1) else dy

        # Pulled along the positive x-direction:
        dof_x = 2 * node_id
        F_ext[dof_x] = traction_mag * segment_length

    return F_ext


def generate_ground_truth(
    cfg: dict,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:

    data_dir = Path(cfg["data"]["data_dir"])
    dataset_name = cfg["data"]["dataset_name"]
    save_file = data_dir / dataset_name
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Domain and Mesh
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    nx = int(cfg["domain"].get("nx", 10))
    ny = int(cfg["domain"].get("ny", 10))
    width = float(x_max - x_min)
    height = float(y_max - y_min)

    # Mesh
    mesh = create_mesh(
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        device="cpu",
    )

    # Temporal Grid
    t0, t_final = cfg["domain"]["t_range"]
    n_steps = int(cfg["domain"]["n_steps"])
    t_eval = torch.linspace(float(t0), float(t_final), n_steps)

    # Physics Constants
    phys_cfg = cfg["physics"]
    sigma_max = float(phys_cfg["sigma_max"])

    # Odgen coefficient
    alphas = torch.tensor(phys_cfg["alphas"], dtype=torch.float64)
    mus_list = [float(mu) for mu in phys_cfg["mus"]]
    mus = torch.tensor(mus_list, dtype=torch.float64)

    # Volumetric parameters
    lam = float(phys_cfg["lambda"])
    beta = float(phys_cfg["beta"])

    # FEM values initialisation
    u = torch.zeros(mesh.n_dofs, dtype=torch.float64)
    dN_dX, areas = triangle_shape_gradients(mesh.nodes, mesh.elements)
    header = f"{'Iter':>5} | {'||R||':>12} | {'||delta_u||':>12} | {'Max |u|':>10}"
    separator = "-" * len(header)

    # Dataset values
    psi_traj = torch.zeros(n_steps, dtype=torch.float64, device=device)
    u_traj = torch.zeros((n_steps, mesh.n_nodes, 2), dtype=torch.float64, device=device)
    F_traj = torch.zeros((n_steps, mesh.n_nodes, 2), dtype=torch.float64, device=device)

    def compute_total_energy(u_flat: torch.Tensor) -> torch.Tensor:
        u = u_flat.view(-1, 2)
        grad_u = compute_u_grad(u, mesh.elements, dN_dX)
        kin = Kinematics(grad_u)
        ogden = Ogden(
            psi=torch.zeros(0),
            mus=mus,
            alphas=alphas,
            beta=beta,
            lam=lam,
            kin=kin,
        )
        psi = ogden.get_2Dpsi()
        if torch.any(kin.J <= 0.0):
            # The mesh has inverted or collapsed in this candidate step
            print(
                f"WARNING: Inverted element detected! Min det(F): {kin.J.min().item():.4e}"
            )
        return torch.sum(psi * areas)

    for n in range(n_steps):  # Newton-Raphson loop
        t_n = t_eval[n]
        F_ext = compute_external_force(
            t_n,
            t_final,
            mesh,
            sigma_max,
            height,
            ny,
        )
        f_ext_norm = torch.norm(F_ext).item()

        print(
            f"\n[Step {n + 1:02d}/{n_steps:02d}] Time: {t_n:.3f} s | ||F_ext||: {f_ext_norm:.4e} N"
        )
        print(separator)
        print(header)
        print(separator)

        nr_iter = 0
        while True:  # Loop interaction
            # Enable the computational graph to use autograd
            current_u = u.detach().clone().requires_grad_(True)

            # Internal force evaluation
            psi_int = compute_total_energy(current_u)
            F_int = torch.autograd.grad(
                outputs=psi_int,
                inputs=current_u,
                grad_outputs=torch.ones_like(psi_int),
                create_graph=True,
            )[0]

            # Equilibrium residual
            Res = F_ext - F_int
            R = Res[mesh.free_dofs]
            res_norm = torch.norm(R).item()

            # Stiffness computation
            K_tot = torch.autograd.functional.hessian(compute_total_energy, current_u)
            K = K_tot[mesh.free_dofs][:, mesh.free_dofs]
            # print("NaN in K_tot:", torch.isnan(K_tot).any().item())

            # Check convergence
            res_norm = torch.norm(R).item()
            if nr_iter == 0:
                res_0 = max(res_norm, 1e-6)

            if (res_norm / res_0 < 1e-4) or (res_norm < 1e-7): # type: ignore
                break

            # Update step
            with torch.no_grad():
                delta_u_free = torch.linalg.solve(K, R)
                alpha = 1.0
                u_trial = u.clone()
                while alpha > 1e-4:
                    u_trial[mesh.free_dofs] = u[mesh.free_dofs] + alpha * delta_u_free
                    psi_trial = compute_total_energy(u_trial)

                    if torch.isfinite(psi_trial) and not torch.isnan(psi_trial):
                        break
                    alpha *= 0.5

                u[mesh.free_dofs] += alpha * delta_u_free

            delta_u_norm = torch.norm(delta_u_free).item()
            u_max = torch.max(torch.abs(u)).item()
            print(
                f"{nr_iter:>5d} | {res_norm:>12.4e} | {delta_u_norm:>12.4e} | {u_max:>10.4e}"
            )

            nr_iter += 1
            if nr_iter > 100:
                raise RuntimeError(
                    f"Newton-Raphson failed to converge at timestep {n} (t = {t_n:.3f} s)."
                )

        psi_traj[n] = psi_int
        u_traj[n] = u.view(-1, 2)
        F_traj[n] = F_ext.view(-1, 2)
    dataset = {
        "time": t_eval.float().to(device),
        "nodes": mesh.nodes.to(device),
        "elements": mesh.elements.to(device),
        "edges": mesh.edges.to(device),
        "boundary_nodes": mesh.boundary_nodes.to(device),
        "left_nodes": mesh.left_nodes.to(device),
        "right_nodes": mesh.right_nodes.to(device),
        "bottom_nodes": mesh.bottom_nodes.to(device),
        "top_nodes": mesh.top_nodes.to(device),
        "u": u_traj,
        "psi": psi_traj,
        "F_ext": F_traj,
    }
    torch.save(dataset, save_file)
    return dataset
