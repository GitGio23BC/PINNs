from pathlib import Path

import torch

from ..geometry import Mesh, create_mesh_3D
from ..physics import Kinematics, Ogden
from ..utils import stencil_grad, tetrahedral_shape_gradients


def compute_lumped_mass_matrix(mesh: Mesh, rho: float, dim: int = 3) -> torch.Tensor:

    nodal_masses = torch.zeros(
        mesh.n_nodes,
        dtype=mesh.nodes.dtype,
        device=mesh.nodes.device,
    )

    Ve = mesh.elements_volume
    element_masses = rho * Ve

    mass_per_corner = element_masses / 4.0

    for i in range(4):
        nodal_masses.scatter_add_(0, mesh.elements[:, i], mass_per_corner)

    M = nodal_masses.repeat_interleave(dim)
    return torch.diag(M)


def compute_external_force(
    t_n: torch.Tensor,
    T: float,
    mesh: Mesh,
    sigma_max: float,
    trac_type: str,
    t0: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = mesh.nodes.dtype
    device = mesh.nodes.device
    F_ext = torch.zeros(mesh.n_dofs, dtype=dtype, device=device)

    # Magnitude (N/m)
    time_value = t_n.to(device=device, dtype=dtype)
    tau = time_value - t0
    if trac_type == "const":
        traction = torch.tensor([sigma_max, 0, 0], dtype=dtype, device=device).to(
            dtype=dtype, device=device
        )
    elif trac_type == "step":
        traction = (
            torch.tensor([sigma_max, 0, 0], dtype=dtype, device=device)
            * (time_value < T / 2)
        ).to(dtype=dtype, device=device)
    elif trac_type == "impulse":
        impulse_duration = T / 10
        traction = (
            torch.tensor([sigma_max, 0, 0], dtype=dtype, device=device)
            * (torch.abs(time_value - T / 2) < impulse_duration / 2)
        ).to(dtype=dtype, device=device)
    elif trac_type == "ramp":
        traction = (
            torch.tensor([sigma_max, 0, 0], dtype=dtype, device=device) * time_value / T
        ).to(dtype=dtype, device=device)
    elif trac_type == "half_sin":
        traction = (
            torch.tensor([sigma_max, 0, 0], dtype=dtype, device=device)
            * torch.sin(torch.pi * time_value / T)
        ).to(dtype=dtype, device=device)
    elif trac_type == "sin":
        traction = (
            torch.tensor([sigma_max, 0, 0], dtype=dtype, device=device)
            * torch.sin(2 * torch.pi * time_value / T)
        ).to(dtype=dtype, device=device)
    elif trac_type == "delayed_step":
        delay = T / 4
        traction = (
            torch.tensor([sigma_max, 0, 0], dtype=dtype, device=device)
            * (tau >= delay)
        ).to(dtype=dtype, device=device)
    else:
        traction = torch.tensor([0, 0, 0], dtype=dtype, device=device)

    right_face_nodes = mesh.get_boundary_faces("right")
    right_face_areas = mesh.get_boundary_areas("right")

    face_nodal_forces = (
        right_face_areas.unsqueeze(1) * traction.unsqueeze(0) / 3.0
    ).repeat_interleave(3, dim=0)

    nodal_forces = torch.zeros_like(mesh.nodes)
    nodal_forces.index_add_(0, right_face_nodes.reshape(-1), face_nodal_forces)
    F_ext = nodal_forces.reshape(-1)

    return F_ext, traction


def generate_inertial_dataset(
    cfg: dict,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    trac_type: str = "const",
    verbose: bool = True,
) -> dict[str, torch.Tensor]:

    data_dir = Path(cfg["data"]["data_dir"])
    dataset_name = str(cfg["data"]["dataset_name"])
    save_file = data_dir / dataset_name
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Domain
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    z_min, z_max = cfg["domain"]["z_range"]
    nx = int(cfg["domain"].get("nx", 10))
    ny = int(cfg["domain"].get("ny", 10))
    nz = int(cfg["domain"].get("nz", 10))
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    depth = float(z_max - z_min)

    # Mesh
    mesh = create_mesh_3D(
        width=width,
        height=height,
        depth=depth,
        nx=nx,
        ny=ny,
        nz=nz,
        device=device,
        dtype=dtype,
    )
    # Temporal Grid
    t0, t_final = cfg["domain"]["t_range"]
    n_steps = int(cfg["domain"]["n_steps"])
    t_eval = torch.linspace(float(t0), float(t_final), n_steps, device=device)

    # Physics Constants
    phys_cfg = cfg["physics"]
    sigma_max = float(phys_cfg["sigma_max"])

    # Odgen coefficient
    tissue = phys_cfg["tissue"]
    alphas_ogden = torch.tensor(phys_cfg["alphas"][tissue], dtype=dtype, device=device)
    mus_list = [float(mu) for mu in phys_cfg["mus"][tissue]]
    mus_ogden = torch.tensor(mus_list, dtype=dtype, device=device)

    # Volumetric parameters
    lambda_ogden = float(phys_cfg["lam"])
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
    # https://mechcodex.com/reference/damping-ratios-typical
    ζ = float(phys_cfg["zeta"])
    ω_min, ω_max = [2.0 * torch.pi * float(f) for f in phys_cfg["f"]]
    ##################
    alpha_rayleigh = 2.0 * ζ * (ω_max + ω_min)
    beta_rayleigh = 2.0 * ζ * (ω_max * ω_min) / (ω_max + ω_min)

    # FEM values initialisation
    u = torch.zeros(
        mesh.n_dofs,
        dtype=dtype,
        device=device,
    )

    dN_dX, volumes = tetrahedral_shape_gradients(mesh.nodes, mesh.elements)
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
        dtype=dtype,
        device=device,
    )
    u_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 3),
        dtype=dtype,
        device=device,
    )
    u_t_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 3),
        dtype=dtype,
        device=device,
    )
    u_tt_traj = torch.zeros(
        (n_steps, mesh.n_nodes, 3),
        dtype=dtype,
        device=device,
    )
    grad_u_traj = torch.zeros(
        (n_steps, mesh.n_elements, 3, 3),
        dtype=dtype,
        device=device,
    )
    u_fixed = torch.zeros(
        (n_steps, *mesh.fixed_dofs.shape),
        dtype=dtype,
        device=device,
    )
    trac_traj = torch.zeros((n_steps, 3), dtype=dtype, device=device)

    # Tolerances
    tol_r_rel = 1e-3
    tol_r_abs = 1e-8

    nr_iter = 0
    max_iters = 100

    # FEM
    def compute_total_energy(u_flat: torch.Tensor) -> torch.Tensor:
        u = u_flat.view(-1, 3)
        grad_u = stencil_grad(u, mesh.elements, dN_dX)
        kin = Kinematics(grad_u)
        ogden = Ogden(
            mus=mus_ogden,
            alphas=alphas_ogden,
            beta=beta_ogden,
            lam=lambda_ogden,
            kin=kin,
        )
        psi = ogden.get_psi()
        if torch.any(kin.J <= 0.0):
            # The mesh has inverted or collapsed in this candidate step
            print(
                f"WARNING: Inverted element detected! Min det(F): {kin.J.min().item():.4e}"
            )
        return torch.sum(psi * volumes)

    for n in range(n_steps):  # Newton-Raphson loop
        t_n = t_eval[n]
        duration = float(t_final - t0)
        F_ext, trac_ext = compute_external_force(
            t_n,
            duration,
            mesh,
            sigma_max,
            trac_type=trac_type,
            t0=float(t0),
        )

        if verbose:
            f_ext_norm = torch.norm(F_ext).item()

            print(
                f"\n[Step {n + 1:02d}/{n_steps:02d}] Time: {t_n:.3f} s | ||F_ext||: {f_ext_norm:.4e} N"
            )
            print(separator)
            print(header)
            print(separator)

        # Take previous step values (note, if n=0 then it will take the last time step, but it is all zeros so all it's fine, do not worry guys, all will work, I hope)
        u_prev = u_traj[n - 1].reshape(-1).to(device)
        u_t_prev = u_t_traj[n - 1].reshape(-1).to(device)
        u_tt_prev = u_tt_traj[n - 1].reshape(-1).to(device)

        nr_iter = 0
        norm_R0 = 0.0
        while True:
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

            K_stiff: torch.Tensor = torch.autograd.functional.hessian(
                compute_total_energy, current_u
            )  # type: ignore
            C = alpha_rayleigh * M + beta_rayleigh * K_stiff

            # Equilibrium residual
            Res = F_ext - F_int - M @ current_u_tt - C @ current_u_t
            R = Res[mesh.free_dofs]
            norm_R = torch.norm(R).item()
            if nr_iter == 0:
                norm_R0 = norm_R
            if not torch.isfinite(R).all() or not torch.isfinite(K_stiff).all():
                raise RuntimeError(
                    f"Non-finite Newton state at time step {n + 1}, "
                    f"iteration {nr_iter}: ||R||={norm_R:.4e}."
                )

            residual_tolerance = max(tol_r_abs, tol_r_rel * norm_R0)
            rel_R = norm_R / max(norm_R0, 1e-12)
            if norm_R <= residual_tolerance:
                break
            if nr_iter >= max_iters:
                raise RuntimeError(
                    f"Newton-Raphson failed to converge at time step {n + 1} "
                    f"(t={t_n.item():.6g}) after {max_iters} iterations: "
                    f"||R||={norm_R:.4e}, relative residual={rel_R:.4e}, "
                    f"tolerance={residual_tolerance:.4e}."
                )

            # Effective stiffness computation
            K_eff = K_stiff + c_a1 * M + c_v1 * C
            K = K_eff[mesh.free_dofs][:, mesh.free_dofs]

            with torch.no_grad():
                delta_u_free = torch.linalg.solve(K, R)
                alpha = 1.0
                accepted = False
                while alpha > 1e-4:
                    u_trial = u.clone()
                    u_trial[mesh.free_dofs] = u[mesh.free_dofs] + alpha * delta_u_free

                    # Freeze damping during backtracking; recompute it next iteration.
                    with torch.enable_grad():
                        trial_u = u_trial.detach().requires_grad_(True)
                        psi_trial = compute_total_energy(trial_u)
                        if torch.isfinite(psi_trial):
                            F_int_trial = torch.autograd.grad(
                                outputs=psi_trial,
                                inputs=trial_u,
                            )[0]
                            u_t_trial = (
                                c_v1 * (trial_u - u_prev)
                                + c_v2 * u_t_prev
                                + c_v3 * u_tt_prev
                            )
                            u_tt_trial = (
                                c_a1 * (trial_u - u_prev)
                                - c_a2 * u_t_prev
                                - c_a3 * u_tt_prev
                            )
                            R_trial = (
                                F_ext
                                - F_int_trial.detach()
                                - M @ u_tt_trial.detach()
                                - C @ u_t_trial.detach()
                            )[mesh.free_dofs]
                            norm_R_trial = torch.norm(R_trial).item()
                            if (
                                torch.isfinite(R_trial).all()
                                and norm_R_trial <= (1.0 - 1e-4 * alpha) * norm_R
                            ):
                                accepted = True
                                break
                    alpha *= 0.5

                if not accepted:
                    raise RuntimeError(
                        f"Newton line search failed at time step {n + 1} "
                        f"(t={t_n.item():.6g}), iteration {nr_iter}: "
                        f"||R||={norm_R:.4e}, last alpha={alpha:.4e}."
                    )
                u.copy_(u_trial)  # type: ignore

            if verbose:
                du_abs = torch.norm(alpha * delta_u_free).item()
                du_rel = du_abs / (torch.norm(u[mesh.free_dofs]).item() + 1e-12)
                print(
                    f"{nr_iter:>4d} | {rel_R:>10.3e} | {norm_R:>10.3e} | {du_rel:>10.3e} | {du_abs:>10.3e} | {alpha:>6.3f}"
                )

            nr_iter += 1

        psi_traj[n] = psi_int.detach()
        u_traj[n] = u.detach().view(-1, 3)
        u_t_traj[n] = current_u_t.detach().view(-1, 3)
        u_tt_traj[n] = current_u_tt.detach().view(-1, 3)
        grad_u_traj[n] = stencil_grad(u.view(-1, 3), mesh.elements, dN_dX)
        trac_traj[n] = trac_ext

    # u_t_traj_n = (u_t_traj - u_t_traj.min()) / (u_t_traj.max() - u_t_traj.min())
    # u_tt_traj_n = (u_tt_traj - u_tt_traj.min()) / (u_tt_traj.max() - u_tt_traj.min())
    # psi_traj_n = (psi_traj - psi_traj.min()) / (psi_traj.max() - psi_traj.min())
    # trac_traj_n = (trac_traj - trac_traj.min()) / (trac_traj.max() - trac_traj.min())

    dataset = {
        "time": t_eval.float().to(device),
        "nodes": mesh.nodes.to(device),
        ###
        "elements": mesh.elements.to(device),
        "edges": mesh.edges.to(device),
        "boundary_nodes": {
            direction: face_node.to(device)
            for direction, face_node in mesh.boundary_nodes.items()
        },
        "left_nodes": mesh.boundary_nodes["left"].to(device),
        "right_nodes": mesh.boundary_nodes["right"].to(device),
        "bottom_nodes": mesh.boundary_nodes["bottom"].to(device),
        "top_nodes": mesh.boundary_nodes["top"].to(device),
        "back_nodes": mesh.boundary_nodes["back"].to(device),
        "front_nodes": mesh.boundary_nodes["front"].to(device),
        "grad_u": grad_u_traj.to(device),
        ###
        "u_t": u_t_traj.to(device),
        "u_tt": u_tt_traj.to(device),
        ###
        "u_fixed": u_fixed.to(device),
        ###
        "u": u_traj.to(device),
        "psi": psi_traj.to(device),
        "trac_ext": trac_traj.to(device),
    }
    torch.save(dataset, save_file)
    return dataset
