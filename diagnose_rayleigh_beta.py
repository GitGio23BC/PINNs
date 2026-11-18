import argparse
from pathlib import Path

import pandas as pd
import torch

from src.geometry import create_graph, create_mesh_3D
from src.loader import MGNData
from src.models import MeshGraphNet
from src.physics import Kinematics, Ogden
from src.physics.operators import compute_internal_nodal_forces
from src.utils import load_config, stencil_grad, tetrahedral_shape_gradients


def rms_free(values: torch.Tensor, free_dofs: torch.Tensor) -> float:
    values_free = values.reshape(-1)[free_dofs]
    return float(values_free.square().mean().sqrt().item())


def cosine_free(
    left: torch.Tensor,
    right: torch.Tensor,
    free_dofs: torch.Tensor,
) -> float:
    left_free = left.reshape(-1)[free_dofs]
    right_free = right.reshape(-1)[free_dofs]
    denominator = (
        torch.linalg.vector_norm(left_free)
        * torch.linalg.vector_norm(right_free)
    )
    if denominator.item() == 0.0:
        return float("nan")
    return float((torch.dot(left_free, right_free) / denominator).item())


def compute_internal_force(
    u: torch.Tensor,
    *,
    mesh,
    dN_dX: torch.Tensor,
    mus: torch.Tensor,
    alphas: torch.Tensor,
    material_beta: float,
    lam: float,
) -> torch.Tensor:
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
    return compute_internal_nodal_forces(P, mesh, dN_dX)


def apply_tangent(
    internal_force: torch.Tensor,
    u: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    # This is the same tangent-vector product used by the Pako beta-damping term.
    return torch.autograd.grad(
        outputs=internal_force,
        inputs=u,
        grad_outputs=velocity,
        create_graph=False,
        retain_graph=True,
    )[0]


def next_available_path(path: Path) -> Path:
    candidate = path
    version = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{version}{path.suffix}")
        version += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare rollout and reference velocities and tangent actions."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint path; defaults to the model configured in config.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
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
        physics["alphas"][tissue],
        device=device,
        dtype=dtype,
    )
    mus = torch.tensor(
        [float(mu) for mu in physics["mus"][tissue]],
        device=device,
        dtype=dtype,
    )
    material_beta = float(physics["beta"])
    lam = float(physics["lam"])

    zeta = float(physics["zeta"])
    w_min, w_max = [
        2.0 * torch.pi * float(frequency)
        for frequency in physics["f"]
    ]
    beta_rayleigh = (
        2.0 * zeta * (w_max * w_min) / (w_max + w_min)
    )

    directions = ("left", "right", "bottom", "top", "back", "front")
    node_type = torch.zeros(
        (mesh.n_nodes, len(directions)),
        device=device,
        dtype=dtype,
    )
    for index, direction in enumerate(directions):
        node_type[mesh.get_boundary_nodes(direction), index] = 1.0

    trac_scale = float(physics["sigma_max"])
    dt = data_loader.dt

    u_prev = torch.zeros((mesh.n_nodes, 3), device=device, dtype=dtype)
    u_prev2 = torch.zeros_like(u_prev)
    velocity_prev = torch.zeros_like(u_prev)
    acceleration_prev = torch.zeros_like(u_prev)
    world_pos = mesh.nodes.clone()

    u_ref_prev = torch.zeros_like(u_prev)
    u_ref_prev2 = torch.zeros_like(u_prev)

    rows: list[dict] = []

    for step in range(data_loader.num_steps):
        batch = data_loader.get_batch(step)
        time_s = float(data_loader.time[step].item())

        trac_feat = torch.zeros(
            (mesh.n_nodes, 3),
            device=device,
            dtype=dtype,
        )
        trac_feat[mesh.get_boundary_nodes("right")] = (
            batch.trac_ext / trac_scale
        )

        graph = create_graph(
            mesh=mesh,
            world_pos=world_pos,
            node_type=node_type,
            velocity=velocity_prev,
            acceleration=acceleration_prev,
            boundary_traction=trac_feat,
            device=device,
            dtype=dtype,
        )

        with torch.enable_grad():
            u_pred = model(graph).to(dtype=dtype)
            u_pred.requires_grad_(True)
            u_t_pred = (u_pred - u_prev) / dt
            u_tt_pred = (u_pred - 2.0 * u_prev + u_prev2) / (dt**2)

            u_ref = data["u"][step].detach().clone().requires_grad_(True)
            u_t_ref_saved = data["u_t"][step].detach()
            u_tt_ref_saved = data["u_tt"][step].detach()

            # Reference derivatives computed with the same finite differences as
            # the predicted rollout, starting from the same zero initial state.
            u_t_ref_fd = (u_ref - u_ref_prev) / dt
            u_tt_ref_fd = (
                u_ref - 2.0 * u_ref_prev + u_ref_prev2
            ) / (dt**2)

            F_int_pred = compute_internal_force(
                u_pred,
                mesh=mesh,
                dN_dX=dN_dX,
                mus=mus,
                alphas=alphas,
                material_beta=material_beta,
                lam=lam,
            )
            F_int_ref = compute_internal_force(
                u_ref,
                mesh=mesh,
                dN_dX=dN_dX,
                mus=mus,
                alphas=alphas,
                material_beta=material_beta,
                lam=lam,
            )

            actions = {
                "pred_tangent_pred_velocity": apply_tangent(
                    F_int_pred, u_pred, u_t_pred
                ),
                "pred_tangent_ref_fd_velocity": apply_tangent(
                    F_int_pred, u_pred, u_t_ref_fd
                ),
                "pred_tangent_ref_saved_velocity": apply_tangent(
                    F_int_pred, u_pred, u_t_ref_saved
                ),
                "ref_tangent_pred_velocity": apply_tangent(
                    F_int_ref, u_ref, u_t_pred
                ),
                "ref_tangent_ref_fd_velocity": apply_tangent(
                    F_int_ref, u_ref, u_t_ref_fd
                ),
                "ref_tangent_ref_saved_velocity": apply_tangent(
                    F_int_ref, u_ref, u_t_ref_saved
                ),
            }

        pred_action = actions["pred_tangent_pred_velocity"]
        ref_action = actions["ref_tangent_ref_saved_velocity"]

        row = {
            "step": step,
            "time_s": time_s,
            "u_t_pred_rms_m_per_s": rms_free(u_t_pred, mesh.free_dofs),
            "u_t_ref_fd_rms_m_per_s": rms_free(u_t_ref_fd, mesh.free_dofs),
            "u_t_ref_saved_rms_m_per_s": rms_free(
                u_t_ref_saved, mesh.free_dofs
            ),
            "u_tt_pred_rms_m_per_s2": rms_free(u_tt_pred, mesh.free_dofs),
            "u_tt_ref_fd_rms_m_per_s2": rms_free(
                u_tt_ref_fd, mesh.free_dofs
            ),
            "u_tt_ref_saved_rms_m_per_s2": rms_free(
                u_tt_ref_saved, mesh.free_dofs
            ),
            "u_t_pred_vs_ref_fd_cosine": cosine_free(
                u_t_pred, u_t_ref_fd, mesh.free_dofs
            ),
            "u_t_pred_vs_ref_saved_cosine": cosine_free(
                u_t_pred, u_t_ref_saved, mesh.free_dofs
            ),
            "u_t_ref_fd_vs_saved_cosine": cosine_free(
                u_t_ref_fd, u_t_ref_saved, mesh.free_dofs
            ),
        }

        for name, action in actions.items():
            damping_force = beta_rayleigh * action
            row[f"{name}_Ku_t_rms_N_per_s"] = rms_free(
                action, mesh.free_dofs
            )
            row[f"{name}_beta_Ku_t_rms_N"] = rms_free(
                damping_force, mesh.free_dofs
            )
            row[f"{name}_cosine_vs_predicted_action"] = cosine_free(
                action, pred_action, mesh.free_dofs
            )
            row[f"{name}_cosine_vs_reference_action"] = cosine_free(
                action, ref_action, mesh.free_dofs
            )

        rows.append(row)

        # Advance the predicted rollout exactly as in test.py.
        u_prev2 = u_prev.detach().clone()
        u_prev = u_pred.detach().clone()
        velocity_prev = ((u_prev - u_prev2) / dt).detach()
        acceleration_prev = u_tt_pred.detach().clone()
        world_pos = (mesh.nodes + u_prev).detach()

        # Advance the finite-difference reference history.
        u_ref_prev2 = u_ref_prev.detach().clone()
        u_ref_prev = u_ref.detach().clone()

    out_dir = Path(cfg.get("output_dir", "./output")) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = next_available_path(
        out_dir / f"{model_file.stem}_rayleigh_beta_comparison.csv"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved velocity/tangent comparison to: {csv_path}")
    print(f"Rayleigh beta coefficient: {beta_rayleigh:.6e} s")


if __name__ == "__main__":
    main()