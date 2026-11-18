from pathlib import Path

import pandas as pd
import torch

from src.geometry import create_mesh_3D
from src.loader import MGNData
from src.loss import pako_loss
from src.physics import Kinematics, Ogden, pako_residual
from src.utils import load_config, stencil_grad, tetrahedral_shape_gradients


def rms_free(values: torch.Tensor, free_dofs: torch.Tensor) -> float:
    values_free = values.reshape(-1)[free_dofs]
    return float(values_free.square().mean().sqrt().item())


def next_available_path(path: Path) -> Path:
    candidate = path
    version = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{version}{path.suffix}")
        version += 1
    return candidate


def main() -> None:
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
    body_force = torch.tensor(
        physics["body_force"],
        device=device,
        dtype=dtype,
    )
    rho = torch.tensor(
        float(physics["rho_0"]),
        device=device,
        dtype=dtype,
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

    # Match the force normalization used in train.py.
    force_reference = torch.zeros(
        (mesh.n_nodes, 3),
        device=device,
        dtype=dtype,
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

    u_prev = torch.zeros((mesh.n_nodes, 3), device=device, dtype=dtype)
    u_prev2 = torch.zeros_like(u_prev)
    dt = data_loader.dt
    rows = []

    for step in range(data_loader.num_steps):
        # The same reference displacement is used for both residuals.
        u = data["u"][step].detach().clone().requires_grad_(True)
        u_t_saved = data["u_t"][step].detach()
        u_tt_saved = data["u_tt"][step].detach()

        # Same backward finite differences and zero-history initialization
        # as the rollout in train.py.
        u_t_fd = (u - u_prev) / dt
        u_tt_fd = (u - 2.0 * u_prev + u_prev2) / (dt**2)

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

        F_ext = data["F_ext"][step]

        residual_saved = pako_residual(
            P=P,
            mesh=mesh,
            dN_dX=dN_dX,
            b=body_force,
            rho=rho,
            u=u,
            u_t=u_t_saved,
            u_tt=u_tt_saved,
            alpha_rayleigh=alpha_rayleigh,
            beta_rayleigh=beta_rayleigh,
            F_ext=F_ext,
        )
        residual_fd = pako_residual(
            P=P,
            mesh=mesh,
            dN_dX=dN_dX,
            b=body_force,
            rho=rho,
            u=u,
            u_t=u_t_fd,
            u_tt=u_tt_fd,
            alpha_rayleigh=alpha_rayleigh,
            beta_rayleigh=beta_rayleigh,
            F_ext=F_ext,
        )

        loss_saved = pako_loss(
            residual=residual_saved,
            force_scale=force_scale,
            free_dofs=mesh.free_dofs,
        )
        loss_fd = pako_loss(
            residual=residual_fd,
            force_scale=force_scale,
            free_dofs=mesh.free_dofs,
        )

        rows.append(
            {
                "step": step,
                "time_s": float(data["time"][step].item()),
                "u_t_saved_rms_m_per_s": rms_free(
                    u_t_saved, mesh.free_dofs
                ),
                "u_t_fd_rms_m_per_s": rms_free(
                    u_t_fd, mesh.free_dofs
                ),
                "u_tt_saved_rms_m_per_s2": rms_free(
                    u_tt_saved, mesh.free_dofs
                ),
                "u_tt_fd_rms_m_per_s2": rms_free(
                    u_tt_fd, mesh.free_dofs
                ),
                "residual_saved_norm_free_N": float(
                    torch.linalg.vector_norm(
                        residual_saved.reshape(-1)[mesh.free_dofs]
                    ).item()
                ),
                "residual_saved_rms_free_N": rms_free(
                    residual_saved, mesh.free_dofs
                ),
                "pako_saved_normalized": float(loss_saved.item()),
                "residual_fd_norm_free_N": float(
                    torch.linalg.vector_norm(
                        residual_fd.reshape(-1)[mesh.free_dofs]
                    ).item()
                ),
                "residual_fd_rms_free_N": rms_free(
                    residual_fd, mesh.free_dofs
                ),
                "pako_fd_normalized": float(loss_fd.item()),
            }
        )

        # Advance the reference displacement history for the next FD step.
        u_prev2 = u_prev.detach().clone()
        u_prev = u.detach().clone()

    output_dir = Path(cfg.get("output_dir", "./output")) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = next_available_path(
        output_dir / f"{data_path.stem}_pako_saved_vs_fd.csv"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    mean_saved = sum(row["pako_saved_normalized"] for row in rows) / len(rows)
    mean_fd = sum(row["pako_fd_normalized"] for row in rows) / len(rows)

    print(f"Saved comparison to: {csv_path}")
    print(f"Mean saved/Newmark Pako loss: {mean_saved:.6e}")
    print(f"Mean finite-difference Pako loss: {mean_fd:.6e}")


if __name__ == "__main__":
    main()