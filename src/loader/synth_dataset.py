from pathlib import Path

import torch

from src.geometry.mesh import Mesh

from ..geometry import create_mesh
from ..physics import Kinematics, Ogden
from ..utils import stencil_grad, triangle_shape_gradients


def compute_lumped_mass_matrix(mesh: Mesh, rho: float, dim: int = 2) -> torch.Tensor:
    elem_coords = mesh.nodes[mesh.elements]
    xe = elem_coords[:, :, 0]
    ye = elem_coords[:, :, 1]

    nodal_masses = torch.zeros(
        mesh.n_nodes,
        dtype=elem_coords.dtype,
        device=elem_coords.device,
    )

    # A = det(x-y)/2
    Ae = 0.5 * (
        xe[:, 0] * (ye[:, 1] - ye[:, 2])
        + xe[:, 1] * (ye[:, 2] - ye[:, 0])
        + xe[:, 2] * (ye[:, 0] - ye[:, 1])
    )

    element_masses = rho * Ae

    mass_per_corner = element_masses / 3.0

    for i in range(3):
        nodal_masses.scatter_add_(0, mesh.elements[:, i], mass_per_corner)

    M = nodal_masses.repeat_interleave(dim)
    return torch.diag(M)


def compute_external_force(
    t_n: torch.Tensor,
    T: float,
    mesh: Mesh,
    sigma_max: float,
    height: float,
    ny: int,
) -> torch.Tensor:
    F_ext = torch.zeros(mesh.n_dofs, dtype=torch.float64, device=mesh.nodes.device)

    # Magnitude (N/m)
    traction_mag = sigma_max * torch.sin(torch.pi * t_n/T)
    # Edge "area"
    dy = height / ny

    # Sorting by y-coordinate
    right_node_indices = mesh.right_nodes
    y_coords = mesh.nodes[right_node_indices, 1]
    sorted_order = torch.argsort(y_coords)
    sorted_right_nodes = right_node_indices[sorted_order]

    num_r_nodes = len(sorted_right_nodes)

    for idx, node_id in enumerate(sorted_right_nodes):
        segment_length = 0.5 * dy if idx in (0, num_r_nodes - 1) else dy

        # Pulled alonge x-direction:
        dof_x = 2 * node_id
        F_ext[dof_x] = traction_mag * segment_length

    return F_ext


def generate_steady_state_dataset(
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
    alphas_ogden = torch.tensor(phys_cfg["alphas"], dtype=torch.float64)
    mus_list = [float(mu) for mu in phys_cfg["mus"]]
    mus_ogden = torch.tensor(mus_list, dtype=torch.float64)

    # Volumetric parameters
    lambda_ogden = float(phys_cfg["lambda"])
    beta_ogden = float(phys_cfg["beta"])

    # FEM values initialisation
    u = torch.zeros(mesh.n_dofs, dtype=torch.float64)
    current_u = torch.zeros(mesh.n_dofs, dtype=torch.float64)

    dN_dX, areas = triangle_shape_gradients(mesh.nodes, mesh.elements)
    header = f"{'Iter':>5} | {'||R||':>12} | {'||delta_u||':>12} | {'Max |u|':>10}"
    separator = "-" * len(header)

    # Dataset values
    psi_traj = torch.zeros(
        n_steps,
        dtype=torch.float64,
        device=device,
    )
    u_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 2),
        dtype=torch.float64,
        device=device,
    )
    grad_u_traj = torch.zeros(
        (n_steps, mesh.n_elements, 2, 2),
        dtype=torch.float64,
        device=device,
    )
    F_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 2),
        dtype=torch.float64,
        device=device,
    )

    # FEM
    def compute_total_energy(u_flat: torch.Tensor) -> torch.Tensor:
        u = u_flat.view(-1, 2)
        grad_u = stencil_grad(u, mesh.elements, dN_dX)
        kin = Kinematics(grad_u)
        ogden = Ogden(
            psi=torch.zeros(0),
            mus=mus_ogden,
            alphas=alphas_ogden,
            beta=beta_ogden,
            lam=lambda_ogden,
            kin=kin,
        )
        psi = ogden._get_2Dpsi()
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
        tollerance = 1e-6
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
            K_stiff = torch.autograd.functional.hessian(compute_total_energy, current_u)
            K = K_stiff[mesh.free_dofs][:, mesh.free_dofs]
            # print("NaN in K_tot:", torch.isnan(K_tot).any().item())

            # Check convergence
            res_norm = torch.norm(R).item()
            if nr_iter == 0:
                res_0 = max(res_norm, tollerance)

            if (res_norm / res_0 < tollerance * 1e2) or (res_norm < tollerance):  # type: ignore
                break

            # Update step
            with torch.no_grad():
                delta_u_free = torch.linalg.solve(K, R)
                alpha = 1.0
                u_trial = u.clone()
                # Secure smooth displacement blend
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
        grad_u_traj[n] = stencil_grad(u.view(-1, 2), mesh.elements, dN_dX)
        F_traj[n] = F_ext.view(-1, 2)
    dataset = {
        "time": t_eval.float().to(device),
        "nodes": mesh.nodes.to(device),
        ###
        "elements": mesh.elements.to(device),
        "edges": mesh.edges.to(device),
        "boundary_nodes": mesh.boundary_nodes.to(device),
        "left_nodes": mesh.left_nodes.to(device),
        "right_nodes": mesh.right_nodes.to(device),
        "bottom_nodes": mesh.bottom_nodes.to(device),
        "top_nodes": mesh.top_nodes.to(device),
        "grad_u": grad_u_traj,
        ###
        "u": u_traj,
        "psi": psi_traj,
        "F_ext": F_traj,
    }
    torch.save(dataset, save_file)
    return dataset


def generate_inertial_dataset(
    cfg: dict,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:

    data_dir = Path(cfg["data"]["data_dir"])
    dataset_name = cfg["data"]["dataset_name"]
    save_file = data_dir / dataset_name
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Domain
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
    alphas_ogden = torch.tensor(phys_cfg["alphas"], dtype=torch.float64)
    mus_list = [float(mu) for mu in phys_cfg["mus"]]
    mus_ogden = torch.tensor(mus_list, dtype=torch.float64)

    # Volumetric parameters
    lambda_ogden = float(phys_cfg["lambda"])
    beta_ogden = float(phys_cfg["beta"])
    rho = float(phys_cfg["rho_0"])
    M = compute_lumped_mass_matrix(mesh, rho)

    # Newmark parameters (https://en.wikipedia.org/wiki/Newmark-beta_method)
    ###################### TO UNDERSTAND IF AVERAGE ACCELARATION IS GOOD
    gamma_newmark = 0.5
    beta_newmark = 0.25
    ######################

    # Rayleigh Parameters (https://www.simscale.com/knowledge-base/rayleigh-damping-coefficients/)
    ################### TO SEARCH
    ζ = 0.03  # https://mechcodex.com/reference/damping-ratios-typical
    ω_max = 1
    ω_min = 0.01
    ##################
    alpha_rayleigh = 2.0 * ζ * (ω_max + ω_min)
    beta_rayleigh = 2.0 * ζ * (ω_max * ω_min) / (ω_max + ω_min)

    # FEM values initialisation
    u = torch.zeros(mesh.n_dofs, dtype=torch.float64)

    dN_dX, areas = triangle_shape_gradients(mesh.nodes, mesh.elements)
    dt = (t_final - t0) / (n_steps - 1)

    # Newmark coefficients
    c_a1 = 1.0 / (beta_newmark * (dt**2))
    c_a2 = 1.0 / (beta_newmark * dt)
    c_a3 = 1.0 / (2.0 * beta_newmark) - 1.0

    c_v1 = gamma_newmark / (beta_newmark * dt)
    c_v2 = 1.0 - gamma_newmark / beta_newmark
    c_v3 = dt * (1.0 - gamma_newmark / (2.0 * beta_newmark))

    header = f"{'Iter':>4} | {'||R_rel||':>10} | {'||R_abs||':>10} | {'||du_rel||':>10} | {'||du_abs||':>10} | {'Alpha':>6}"
    separator = "-" * len(header)

    # Dataset values
    psi_traj = torch.zeros(
        n_steps,
        dtype=torch.float64,
        device=device,
    )
    u_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 2),
        dtype=torch.float64,
        device=device,
    )
    u_t_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 2),
        dtype=torch.float64,
        device=device,
    )
    u_tt_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 2),
        dtype=torch.float64,
        device=device,
    )
    grad_u_traj = torch.zeros(
        (n_steps, mesh.n_elements, 2, 2),
        dtype=torch.float64,
        device=device,
    )
    F_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 2),
        dtype=torch.float64,
        device=device,
    )

    # Tolerances
    tol_r_rel = 1e-5
    tol_r_abs = 1e-8
    tol_u_rel = 1e-4
    tol_u_abs = 1e-7

    nr_iter = 0
    max_iters = 100

    # FEM
    def compute_total_energy(u_flat: torch.Tensor) -> torch.Tensor:
        u = u_flat.view(-1, 2)
        grad_u = stencil_grad(u, mesh.elements, dN_dX)
        kin = Kinematics(grad_u)
        ogden = Ogden(
            psi=torch.zeros(0),
            mus=mus_ogden,
            alphas=alphas_ogden,
            beta=beta_ogden,
            lam=lambda_ogden,
            kin=kin,
        )
        psi = ogden._get_2Dpsi()
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

        # Take previous step values (note, if n=0 then it will take the last time step, but it is all zeros so all it's fine, do not worry guys, all will work, I hope)
        u_prev = u_traj[n - 1].reshape(-1)
        u_t_prev = u_t_traj[n - 1].reshape(-1)
        u_tt_prev = u_tt_traj[n - 1].reshape(-1)

        nr_iter = 0
        norm_R0 = 0
        delta_u_free = 0
        while True:  # Loop interaction
            # Enable the computational graph to use autograd
            current_u = u.detach().clone().requires_grad_(True)
            current_u_t = (
                c_v1 * (current_u - u_prev) + c_v2 * u_t_prev + c_v3 * u_tt_prev
            )
            current_u_tt = (
                c_a1 * (current_u - u_prev) - c_a2 * u_t_prev - c_a3 * u_tt_prev
            )

            # Internal force evaluation
            psi_int = compute_total_energy(current_u)
            F_int = torch.autograd.grad(
                outputs=psi_int,
                inputs=current_u,
                grad_outputs=torch.ones_like(psi_int),
                create_graph=True,
            )[0]

            #  Coefficeint compute
            K_stiff = torch.autograd.functional.hessian(compute_total_energy, current_u)
            C = alpha_rayleigh * M + beta_rayleigh * K_stiff  # type: ignore

            # Equilibrioum residual
            Res = F_ext - F_int - M @ current_u_tt - C @ current_u_t
            R = Res[mesh.free_dofs]

            # Check convergence
            norm_R = torch.norm(R).item()
            norm_u = torch.norm(u).item()
            if nr_iter == 0:
                norm_R0 = norm_R
            rel_R = norm_R / (norm_R0 + 1e-12)

            # Effetive Stiffness computation
            K_eff = K_stiff + c_a1 * M + c_v1 * C
            K = K_eff[mesh.free_dofs][:, mesh.free_dofs]
            # print("NaN in K_tot:", torch.isnan(K_tot).any().item())

            # Update step
            with torch.no_grad():
                delta_u_free = torch.linalg.solve(K, R)
                alpha = 1.0
                u_trial = u.clone()
                # Secure smooth displacement blend
                while alpha > 1e-4:
                    u_trial[mesh.free_dofs] = u[mesh.free_dofs] + alpha * delta_u_free
                    psi_trial = compute_total_energy(u_trial)

                    if torch.isfinite(psi_trial) and not torch.isnan(psi_trial):
                        break
                    alpha *= 0.5

                u[mesh.free_dofs] += alpha * delta_u_free

            # Loggig
            du_abs = torch.norm(alpha * delta_u_free).item()
            du_rel = du_abs / (torch.norm(u[mesh.free_dofs]).item() + 1e-12)
            print(
                f"{nr_iter:>4d} | {rel_R:>10.3e} | {norm_R:>10.3e} | {du_rel:>10.3e} | {du_abs:>10.3e} | {alpha:>6.3f}"
            )
            nr_iter += 1

            if nr_iter > 0:
                norm_du = torch.norm(delta_u_free).item()
                rel_du = norm_du / (norm_u + 1e-12)

                r_converged = (rel_R < tol_r_rel) or (norm_R < tol_r_abs)
                u_converged = (rel_du < tol_u_rel) or (norm_du < tol_u_abs)
                if r_converged and u_converged:
                    break
            if nr_iter >= max_iters:
                raise RuntimeError(
                    f"Newton-Raphson failed to converge after {max_iters} iterations."
                )

        psi_traj[n] = psi_int
        u_traj[n] = u.detach().view(-1, 2)
        u_t_traj[n] = current_u_t.detach().view(-1, 2)
        u_tt_traj[n] = current_u_tt.detach().view(-1, 2)
        grad_u_traj[n] = stencil_grad(u.view(-1, 2), mesh.elements, dN_dX)
        F_traj[n] = F_ext.view(-1, 2)

    dataset = {
        "time": t_eval.float().to(device),
        "nodes": mesh.nodes.to(device),
        ###
        "elements": mesh.elements.to(device),
        "edges": mesh.edges.to(device),
        "boundary_nodes": mesh.boundary_nodes.to(device),
        "left_nodes": mesh.left_nodes.to(device),
        "right_nodes": mesh.right_nodes.to(device),
        "bottom_nodes": mesh.bottom_nodes.to(device),
        "top_nodes": mesh.top_nodes.to(device),
        "grad_u": grad_u_traj,
        ###
        "u": u_traj,
        "u_t": u_t_traj,
        "u_tt": u_tt_traj,
        "psi": psi_traj,
        "F_ext": F_traj,
    }
    torch.save(dataset, save_file)
    return dataset
