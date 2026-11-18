import argparse
from pathlib import Path

import pandas as pd
import torch

from src.geometry import create_graph, create_mesh_3D
from src.loader import MGNData
from src.models import MeshGraphNet
from src.physics import Kinematics, Ogden, pako_residual
from src.physics.operators import (
    compute_internal_nodal_forces,
    compute_nodal_masses,
)
from src.utils import load_config, stencil_grad, tetrahedral_shape_gradients


def compute_signed_components(
    u: torch.Tensor,
    u_t: torch.Tensor,
    u_tt: torch.Tensor,
    F_ext: torch.Tensor,
    *,
    mesh,
    dN_dX: torch.Tensor,
    mus: torch.Tensor,
    alphas: torch.Tensor,
    material_beta: float,
    lam: float,
    rho: torch.Tensor,
    body_force: torch.Tensor,
    alpha_rayleigh: float,
    beta_rayleigh: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    grad_u = stencil_grad(u, mesh.elements, dN_dX)
    kin = Kinematics(grad_u)
    ogden = Ogden(
        mus=mus,
        alphas=alphas,
        beta=material_beta,
        lam=lam,
        kin=kin,
    )
    ogden.get_psi()
    P = ogden.get_P()

    internal = compute_internal_nodal_forces(P, mesh, dN_dX)
    nodal_mass = compute_nodal_masses(mesh, rho)
    inertia = nodal_mass * u_tt
    damping_mass = alpha_rayleigh * nodal_mass * u_t

    damping_stiffness = torch.zeros_like(internal)
    if beta_rayleigh != 0.0:
        stiffness_velocity = torch.autograd.grad(
            outputs=internal,
            inputs=u,
            grad_outputs=u_t,
            create_graph=True,
            retain_graph=True,
        )[0]
        damping_stiffness = beta_rayleigh * stiffness_velocity

    body = -(nodal_mass * body_force.unsqueeze(0))
    external = -F_ext

    components = {
        "internal": internal,
        "inertia": inertia,
        "damping_mass": damping_mass,
        "damping_stiffness": damping_stiffness,
        "body": body,
        "external": external,
    }

    component_sum = torch.zeros_like(internal)
    for component in components.values():
        component_sum = component_sum + component

    residual = pako_residual(
        P=P,
        mesh=mesh,
        dN_dX=dN_dX,
        b=body_force,
        rho=rho,
        u=u,
        u_t=u_t,
        u_tt=u_tt,
        alpha_rayleigh=alpha_rayleigh,
        beta_rayleigh=beta_rayleigh,
        F_ext=F_ext,
    )

    return components, residual, component_sum


def append_diagnostics(
    rows: list[dict],
    *,
    trajectory: str,
    step: int,
    time_s: float,
    components: dict[str, torch.Tensor],
    residual: torch.Tensor,
    component_sum: torch.Tensor,
    force_scale: torch.Tensor,
    free_dofs: torch.Tensor,
) -> None:
    residual_free = residual.reshape(-1)[free_dofs]
    residual_energy = residual_free.square().sum()

    sum_error = (component_sum - residual).reshape(-1)[free_dofs]
    component_sum_error_over_force_rms = (
        sum_error.square().mean().sqrt() / force_scale.sqrt()
    ).item()

    for name, component in components.items():
        component_free = component.reshape(-1)[free_dofs]
        projection_share = float("nan")

        if residual_energy.item() > 0.0:
            projection_share = (
                torch.dot(component_free, residual_free) / residual_energy
            ).item()

        rows.append(
            {
                "trajectory": trajectory,
                "step": step,
                "time_s": time_s,
                "term": name,
                "norm_free_N": torch.linalg.vector_norm(component_free).item(),
                "rms_free_N": component_free.square().mean().sqrt().item(),
                "normalized_mean_square": (
                    component_free.square().mean() / force_scale
                ).item(),
                "projection_on_total": projection_share,
                "component_sum_error_over_force_rms": (
                    component_sum_error_over_force_rms
                ),
            }
        )

    rows.append(
        {
            "trajectory": trajectory,
            "step": step,
            "time_s": time_s,
            "term": "total_residual",
            "norm_free_N": torch.linalg.vector_norm(residual_free).item(),
            "rms_free_N": residual_free.square().mean().sqrt().item(),
            "normalized_mean_square": (
                residual_free.square().mean() / force_scale
            ).item(),
            "projection_on_total": 1.0 if residual_energy.item() > 0.0 else float("nan"),
            "component_sum_error_over_force_rms": (
                component_sum_error_over_force_rms
            ),
        }
    )


def next_available_path(path: Path) -> Path:
    candidate = path
    version = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{version}{path.suffix}")
        version += 1
    return candidate


def append_kinematic_diagnostics(
    rows: list[dict],
    *,
    rollout: str,
    step: int,
    time_s: float,
    first_loaded_step: int,
    force_norm: torch.Tensor,
    u: torch.Tensor,
    u_t: torch.Tensor,
    u_tt: torch.Tensor,
    u_prev: torch.Tensor,
    velocity_prev: torch.Tensor,
    acceleration_prev: torch.Tensor,
    u_reference: torch.Tensor,
) -> None:
    rows.append(
        {
            "rollout": rollout,
            "step": step,
            "time_s": time_s,
            "before_first_load": step < first_loaded_step,
            "force_norm_N": force_norm.item(),
            "u_rms_m": u.square().mean().sqrt().item(),
            "u_max_abs_m": u.abs().max().item(),
            "u_reference_rms_m": u_reference.square().mean().sqrt().item(),
            "u_error_rms_m": (u - u_reference).square().mean().sqrt().item(),
            "displacement_increment_rms_m": (
                u - u_prev
            ).square().mean().sqrt().item(),
            "velocity_prev_rms_m_per_s": (
                velocity_prev.square().mean().sqrt().item()
            ),
            "acceleration_prev_rms_m_per_s2": (
                acceleration_prev.square().mean().sqrt().item()
            ),
            "u_t_rms_m_per_s": u_t.square().mean().sqrt().item(),
            "u_t_max_abs_m_per_s": u_t.abs().max().item(),
            "u_tt_rms_m_per_s2": u_tt.square().mean().sqrt().item(),
            "u_tt_max_abs_m_per_s2": u_tt.abs().max().item(),
        }
    )


def main() -> None:
    cfg = load_config("config.yaml")

    parser = argparse.ArgumentParser(
        description="Decompose Pako momentum residual on prediction and reference."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint path; defaults to the model configured in config.yaml.",
    )
    args = parser.parse_args()

    device = torch.device(cfg["training"].get("device", "cuda"))
    dtype = torch.float32

    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    z_min, z_max = cfg["domain"]["z_range"]

    mesh = create_mesh_3D(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        depth=float(z_max - z_min),
        nx=int(cfg["domain"]["nx"]),
        ny=int(cfg["domain"]["ny"]),
        nz=int(cfg["domain"]["nz"]),
        device=device,
        dtype=dtype,
    )
    dN_dX, _ = tetrahedral_shape_gradients(mesh.nodes, mesh.elements)

    data_path = Path(cfg["data"]["data_dir"]) / cfg["data"]["dataset_name"]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = torch.load(data_path, map_location=device, weights_only=False)
    data_loader = MGNData(
        dataset=dataset,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )
    data = data_loader.dataset

    required = ("u", "u_t", "u_tt", "trac_ext")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset missing required fields: {missing}")

    traction_norm = torch.linalg.vector_norm(data["trac_ext"], dim=-1)
    loaded_steps = torch.nonzero(traction_norm > 0.0, as_tuple=False).flatten()
    if loaded_steps.numel() == 0:
        raise ValueError("Dataset contains no non-zero external traction.")
    first_loaded_step = int(loaded_steps[0].item())

    model_cfg = cfg["model"]
    default_model = (
        Path(cfg["output_dir"])
        / str(model_cfg["model_name"])
        / f"{model_cfg['model_name']}.pt"
    )
    model_file = args.model or default_model
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state_dict)
    model.eval()

    physics = cfg["physics"]
    tissue = physics["tissue"]
    alphas = torch.tensor(
        physics["alphas"][tissue], device=device, dtype=dtype
    )
    mus = torch.tensor(
        [float(mu) for mu in physics["mus"][tissue]],
        device=device,
        dtype=dtype,
    )
    material_beta = float(physics["beta"])
    lam = float(physics["lam"])
    body_force = torch.tensor(
        physics["body_force"], device=device, dtype=dtype
    )
    if first_loaded_step > 0:
        unloaded_displacements = data["u"][:first_loaded_step]
        if torch.count_nonzero(body_force).item() != 0:
            raise ValueError(
                "The pre-load reset comparison requires zero body force."
            )
        if not torch.allclose(
            unloaded_displacements,
            torch.zeros_like(unloaded_displacements),
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "The pre-load reset comparison requires zero reference "
                "displacements before the first applied traction."
            )

    rho = torch.tensor(
        [float(physics["rho_0"])], device=device, dtype=dtype
    )

    zeta = float(physics["zeta"])
    w_min, w_max = [
        2.0 * torch.pi * float(frequency)
        for frequency in physics["f"]
    ]
    alpha_rayleigh = 2.0 * zeta * (w_max + w_min)
    beta_rayleigh = (
        2.0 * zeta * (w_max * w_min) / (w_max + w_min)
    )

    force_reference = torch.zeros(
        (mesh.n_nodes, 3), device=device, dtype=dtype
    )
    right_nodes = mesh.get_boundary_nodes("right")
    right_areas = mesh.get_boundary_nodal_areas("right")
    force_reference[right_nodes, 0] = (
        float(physics["sigma_max"]) * right_areas[right_nodes]
    )
    force_scale = (
        force_reference.reshape(-1)[mesh.free_dofs]
        .square()
        .mean()
        .clamp_min(1e-8)
        .detach()
    )

    directions = ("left", "right", "bottom", "top", "back", "front")
    node_type = torch.zeros(
        (mesh.n_nodes, len(directions)), device=device, dtype=dtype
    )
    for index, direction in enumerate(directions):
        node_type[mesh.get_boundary_nodes(direction), index] = 1.0

    trac_scale = float(physics["sigma_max"])
    dt = data_loader.dt
    u_prev = torch.zeros((mesh.n_nodes, 3), device=device, dtype=dtype)
    velocity_prev = torch.zeros_like(u_prev)
    acceleration_prev = torch.zeros_like(u_prev)
    world_pos = mesh.nodes.clone()

    u_prev_reset = torch.zeros_like(u_prev)
    velocity_prev_reset = torch.zeros_like(u_prev)
    acceleration_prev_reset = torch.zeros_like(u_prev)
    world_pos_reset = mesh.nodes.clone()

    rows: list[dict] = []
    kinematic_rows: list[dict] = []

    for step in range(data_loader.num_steps):
        batch = data_loader.get_batch(step)
        time_s = float(data_loader.time[step].item())

        trac_feat = torch.zeros(
            (mesh.n_nodes, 3), device=device, dtype=dtype
        )
        trac_feat[mesh.get_boundary_nodes("right")] = (
            batch.trac_ext / trac_scale
        )

        graph = create_graph(
            mesh=mesh,
            world_pos=world_pos,
            node_type=node_type,
            velocity=velocity_prev,
            boundary_traction=trac_feat,
            device=device,
            dtype=dtype,
        )

        graph_reset = create_graph(
            mesh=mesh,
            world_pos=world_pos_reset,
            node_type=node_type,
            velocity=velocity_prev_reset,
            boundary_traction=trac_feat,
            device=device,
            dtype=dtype,
        )

        with torch.enable_grad():
            u_pred = model(graph).to(dtype=dtype)
            u_pred.requires_grad_(True)
            u_t_pred = 2.0 / dt * (u_pred - u_prev) - velocity_prev
            u_tt_pred = (
                4.0 / (dt**2) * (u_pred - u_prev)
                - 4.0 / dt * velocity_prev
                - acceleration_prev
            )

            pred_components, pred_residual, pred_sum = compute_signed_components(
                u_pred,
                u_t_pred,
                u_tt_pred,
                batch.F_ext,
                mesh=mesh,
                dN_dX=dN_dX,
                mus=mus,
                alphas=alphas,
                material_beta=material_beta,
                lam=lam,
                rho=rho,
                body_force=body_force,
                alpha_rayleigh=alpha_rayleigh,
                beta_rayleigh=beta_rayleigh,
            )

            u_ref = data["u"][step].detach().clone().requires_grad_(True)
            u_t_ref = data["u_t"][step].detach()
            u_tt_ref = data["u_tt"][step].detach()

            ref_components, ref_residual, ref_sum = compute_signed_components(
                u_ref,
                u_t_ref,
                u_tt_ref,
                batch.F_ext,
                mesh=mesh,
                dN_dX=dN_dX,
                mus=mus,
                alphas=alphas,
                material_beta=material_beta,
                lam=lam,
                rho=rho,
                body_force=body_force,
                alpha_rayleigh=alpha_rayleigh,
                beta_rayleigh=beta_rayleigh,
            )

            utt_ref_components, utt_ref_residual, utt_ref_sum = (
                compute_signed_components(
                    u_pred,
                    u_t_pred,
                    u_tt_ref,
                    batch.F_ext,
                    mesh=mesh,
                    dN_dX=dN_dX,
                    mus=mus,
                    alphas=alphas,
                    material_beta=material_beta,
                    lam=lam,
                    rho=rho,
                    body_force=body_force,
                    alpha_rayleigh=alpha_rayleigh,
                    beta_rayleigh=beta_rayleigh,
                )
            )

            ut_ref_components, ut_ref_residual, ut_ref_sum = (
                compute_signed_components(
                    u_pred,
                    u_t_ref,
                    u_tt_pred,
                    batch.F_ext,
                    mesh=mesh,
                    dN_dX=dN_dX,
                    mus=mus,
                    alphas=alphas,
                    material_beta=material_beta,
                    lam=lam,
                    rho=rho,
                    body_force=body_force,
                    alpha_rayleigh=alpha_rayleigh,
                    beta_rayleigh=beta_rayleigh,
                )
            )

        with torch.no_grad():
            u_reset = model(graph_reset).to(dtype=dtype)
            u_t_reset = (
                2.0 / dt * (u_reset - u_prev_reset) - velocity_prev_reset
            )
            u_tt_reset = (
                4.0 / (dt**2) * (u_reset - u_prev_reset)
                - 4.0 / dt * velocity_prev_reset
                - acceleration_prev_reset
            )

        append_diagnostics(
            rows,
            trajectory="predicted",
            step=step,
            time_s=time_s,
            components=pred_components,
            residual=pred_residual,
            component_sum=pred_sum,
            force_scale=force_scale,
            free_dofs=mesh.free_dofs,
        )

        u_reference = data["u"][step]
        force_norm = torch.linalg.vector_norm(batch.F_ext)
        append_kinematic_diagnostics(
            kinematic_rows,
            rollout="autoregressive",
            step=step,
            time_s=time_s,
            first_loaded_step=first_loaded_step,
            force_norm=force_norm,
            u=u_pred.detach(),
            u_t=u_t_pred.detach(),
            u_tt=u_tt_pred.detach(),
            u_prev=u_prev,
            velocity_prev=velocity_prev,
            acceleration_prev=acceleration_prev,
            u_reference=u_reference,
        )
        append_kinematic_diagnostics(
            kinematic_rows,
            rollout="reset_state_before_first_load",
            step=step,
            time_s=time_s,
            first_loaded_step=first_loaded_step,
            force_norm=force_norm,
            u=u_reset,
            u_t=u_t_reset,
            u_tt=u_tt_reset,
            u_prev=u_prev_reset,
            velocity_prev=velocity_prev_reset,
            acceleration_prev=acceleration_prev_reset,
            u_reference=u_reference,
        )

        append_diagnostics(
            rows,
            trajectory="predicted_u_tt_reference",
            step=step,
            time_s=time_s,
            components=utt_ref_components,
            residual=utt_ref_residual,
            component_sum=utt_ref_sum,
            force_scale=force_scale,
            free_dofs=mesh.free_dofs,
        )
        append_diagnostics(
            rows,
            trajectory="predicted_u_t_reference",
            step=step,
            time_s=time_s,
            components=ut_ref_components,
            residual=ut_ref_residual,
            component_sum=ut_ref_sum,
            force_scale=force_scale,
            free_dofs=mesh.free_dofs,
        )
        append_diagnostics(
            rows,
            trajectory="reference",
            step=step,
            time_s=time_s,
            components=ref_components,
            residual=ref_residual,
            component_sum=ref_sum,
            force_scale=force_scale,
            free_dofs=mesh.free_dofs,
        )

        u_prev = u_pred.detach().clone()
        velocity_prev = u_t_pred.detach().clone()
        acceleration_prev = u_tt_pred.detach().clone()
        world_pos = (mesh.nodes + u_prev).detach()

        if step >= first_loaded_step:
            u_prev_reset = u_reset.detach().clone()
            velocity_prev_reset = u_t_reset.detach().clone()
            acceleration_prev_reset = u_tt_reset.detach().clone()
            world_pos_reset = (mesh.nodes + u_prev_reset).detach()

    out_dir = Path(cfg.get("output_dir", "./output")) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = next_available_path(
        out_dir / f"{model_file.stem}_pako_components.csv"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    max_sum_error = max(
        row["component_sum_error_over_force_rms"] for row in rows
    )
    print(f"Saved component diagnostics to: {csv_path}")
    print(
        "Maximum component-sum discrepancy / characteristic force RMS: "
        f"{max_sum_error:.3e}"
    )
    kinematic_path = next_available_path(
        out_dir / f"{model_file.stem}_preload_kinematics.csv"
    )
    pd.DataFrame(kinematic_rows).to_csv(kinematic_path, index=False)
    print(f"Saved pre-load kinematic diagnostics to: {kinematic_path}")
    print(f"First loaded step: {first_loaded_step} (t={data_loader.time[first_loaded_step].item():.6g} s)")


if __name__ == "__main__":
    main()