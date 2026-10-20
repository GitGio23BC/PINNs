import logging
import re
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.geometry import create_graph, create_mesh
from src.loader import MGNData, generate_inertial_dataset
from src.loss import rl2e
from src.models import MeshGraphNet
from src.physics import Kinematics
from src.physics.equations import Ogden
from src.physics.operators import pako_residual_2D
from src.utils import init_logging, load_config
from src.utils.tensor_tools import stencil_grad, triangle_shape_gradients


def _free_region_masks(mesh) -> dict[str, torch.Tensor]:
    """Return disjoint node regions, represented on the flattened free DOFs."""
    n_nodes = mesh.n_nodes
    boundary = torch.zeros(n_nodes, dtype=torch.bool, device=mesh.nodes.device)
    boundary[mesh.boundary_nodes] = True

    right = torch.zeros_like(boundary)
    right[mesh.right_nodes] = True
    left = torch.zeros_like(boundary)
    left[mesh.left_nodes] = True
    top = torch.zeros_like(boundary)
    top[mesh.top_nodes] = True
    bottom = torch.zeros_like(boundary)
    bottom[mesh.bottom_nodes] = True

    regions = {
        "interior": ~boundary,
        "right_loaded": right,
        "left_support_free": left & ~right,
        "top_bottom": (top | bottom) & ~right & ~left,
    }
    free_dofs = mesh.free_dofs
    masks = {}
    for name, node_mask in regions.items():
        dof_mask = node_mask.unsqueeze(-1).expand(-1, 2).reshape(-1)
        masks[name] = free_dofs[torch.isin(free_dofs, torch.nonzero(dof_mask).flatten())]
    return masks


def test(model_path: Path | None = None):
    # Logging and set-up
    init_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config()
    requested_device = cfg["training"].get("device", "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "È stata richiesta una GPU CUDA per il test, ma PyTorch non la rileva."
        )
    device = torch.device(requested_device)
    dtype = torch.float32
    output_dir = Path(cfg.get("output_dir", "./output")) / cfg['model']['model_name']

    if model_path is None:
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"
    model_name = model_path.stem

    # Mesh
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        nx=nx,
        ny=ny,
        device=device,
    )

    dN_dX, _ = triangle_shape_gradients(mesh.nodes, mesh.elements)

    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)
    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1
    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    dataset = {
        key: value.to(device=device, dtype=dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value.to(device=device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in dataset.items()
    }

    data_loader = MGNData(dataset, mesh)
    time_steps = len(dataset["time"])
    dt = data_loader.dt
    time_grid = dataset["time"]
    traction_scale = float(cfg["physics"]["sigma_max"])
    region_dofs = _free_region_masks(mesh)
    force_reference = torch.zeros(
        (mesh.n_nodes, 2), dtype=dtype, device=device
    )
    force_reference[mesh.right_nodes, 0] = (
        traction_scale * mesh.right_lengths
    )
    force_scale_free = torch.mean(
        force_reference.reshape(-1)[mesh.free_dofs] ** 2
    ).clamp_min(1e-8)
    nonzero_force = force_reference.reshape(-1)[mesh.free_dofs].abs() > 0
    force_scale_nonzero = torch.mean(
        force_reference.reshape(-1)[mesh.free_dofs][nonzero_force] ** 2
    ).clamp_min(1e-8)
    logger.info(
        "PAKO force scales: free_mean_sq=%.6e, nonzero_mean_sq=%.6e, "
        "free_dofs=%d, nonzero_force_dofs=%d",
        force_scale_free.item(),
        force_scale_nonzero.item(),
        mesh.free_dofs.numel(),
        nonzero_force.sum().item(),
    )

    # Model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_cfg = cfg["model"]
    layer_indices = [
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"processor\.layers\.(\d+)\.", key))
    ]
    num_layers = max(layer_indices, default=int(model_cfg["num_layers"])) + 1
    mgn = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=num_layers,
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=dtype)
    mgn.load_state_dict(state_dict)
    mgn.eval()
    logger.info(f"Loaded trained MeshGraphNet model from {model_path}")

    # Physics Constants
    p_cfg = cfg["physics"]
    tissue = p_cfg["tissue"]
    alphas = torch.tensor(p_cfg["alphas"][tissue], device=device, dtype=dtype)
    mus_list = [float(mu) for mu in p_cfg["mus"][tissue]]
    mus = torch.tensor(mus_list, device=device, dtype=dtype)
    lam = float(p_cfg["lam"])
    beta = float(p_cfg["beta"])
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=dtype)
    rho = float(p_cfg["rho_0"])

    # Rollout Initialisation
    u_prev = torch.zeros((mesh.n_nodes, 2), dtype=dtype, device=device)
    u_prev2 = torch.zeros((mesh.n_nodes, 2), dtype=dtype, device=device)
    velocity_prev = torch.zeros_like(u_prev)
    world_pos = mesh.nodes.clone()

    all_u_pred = []
    all_S_pred = []
    all_pako_res = []

    # Diagnostics arrays for stress-strain analysis
    E11_pred_traj = []
    S11_pred_traj = []
    E11_gt_traj = []
    S11_gt_traj = []

    right_elems = mesh.right_elements

    # Rollout Testing
    logger.info("Starting PINN test rollout...")
    for t_step in range(time_steps):
        batch = data_loader.get_batch(t_step)
        trac_feat = torch.zeros((mesh.n_nodes, 2), device=device, dtype=dtype)
        trac_feat[mesh.right_nodes] =batch.trac_ext / traction_scale
        velocity_feat = velocity_prev

        graph = create_graph(
            mesh=mesh,
            world_pos=world_pos,
            node_type=node_type,
            velocity=velocity_feat,
            boundary_traction= trac_feat,
            device=device,
        )

        with torch.enable_grad():
            u = mgn(graph).to(dtype=dtype)
            u.requires_grad_(True)

            # Acceleration & External Force
            u_tt = (u - 2.0 * u_prev + u_prev2) / (dt**2)
            F_ext = torch.zeros((mesh.n_nodes, 2), dtype=dtype, device=device)
            if batch.trac_ext is not None:
                F_ext[mesh.right_nodes] = batch.trac_ext * mesh.right_lengths.unsqueeze(
                    -1
                )

            # Kinematics & Stress (Prediction)
            grad_u = stencil_grad(u, mesh.elements, dN_dX)
            kin = Kinematics(grad_u)
            ogden = Ogden(mus=mus, alphas=alphas, beta=beta, lam=lam, kin=kin)
            ogden.get_2Dpsi()
            S = ogden.get_S()
            P = kin.compute_P(S)

            # Momentum Residual
            R_pako = pako_residual_2D(P, mesh, dN_dX, b, rho, u_tt, F_ext)

            # Kinematics & Stress (Refernce)
            u_exact = batch.u_exact.to(device=device).to(dtype).detach().clone()
            u_exact.requires_grad_(True)
            grad_u_exact = stencil_grad(u_exact, mesh.elements, dN_dX)
            kin_exact = Kinematics(grad_u_exact)
            ogden_exact = Ogden(
                mus=mus, alphas=alphas, beta=beta, lam=lam, kin=kin_exact
            )
            ogden_exact.get_2Dpsi()
            S_exact = ogden_exact.get_S()

        e11_pred = kin.E[right_elems, 0, 0].mean().item()
        s11_pred = S[right_elems, 0, 0].mean().item()
        e11_gt = kin_exact.E[right_elems, 0, 0].mean().item()
        s11_gt = S_exact[right_elems, 0, 0].mean().item()

        E11_pred_traj.append(e11_pred)
        S11_pred_traj.append(s11_pred)
        E11_gt_traj.append(e11_gt)
        S11_gt_traj.append(s11_gt)

        all_u_pred.append(u.detach().cpu())
        all_S_pred.append(S.detach().cpu())
        all_pako_res.append(R_pako.reshape(-1)[mesh.free_dofs].detach().cpu())

        u_prev2 = u_prev.detach().clone()
        u_prev = u.detach().clone()
        velocity_prev = ((u_prev - u_prev2) / dt).detach()
        world_pos = (mesh.nodes + u_prev).detach()

    # Metrics computation
    reference_trajectory = dataset["u_complete"].detach().cpu()
    reference_norm = torch.linalg.norm(reference_trajectory.flatten()).clamp_min(1e-8)
    l2_err_traj = [
        torch.linalg.norm(u_pred - reference_trajectory[t]).div(reference_norm).item()
        for t, u_pred in enumerate(all_u_pred)
    ]
    final_l2_error = l2_err_traj[-1]
    mean_pako = torch.cat(all_pako_res).abs().mean().item()
    residual_free = torch.cat(all_pako_res)
    region_summary = []
    for name, dofs in region_dofs.items():
        if dofs.numel() == 0:
            continue
        positions = torch.searchsorted(
            mesh.free_dofs.detach().cpu(), dofs.detach().cpu()
        )
        values = residual_free.reshape(time_steps, -1)[:, positions]
        region_summary.append(
            f"{name}: mean_abs={values.abs().mean().item():.6e}, "
            f"rms={torch.sqrt(torch.mean(values**2)).item():.6e}, "
            f"max_abs={values.abs().max().item():.6e}"
        )
    logger.info("Free-DOF momentum residuals by region | %s", " | ".join(region_summary))

    logger.info("=" * 60)
    logger.info(f"Evaluation Results for: {model_path.name}")
    logger.info(f"Final Relative L2 Displacement Error: {final_l2_error:.2%}")
    logger.info(f"Mean Trajectory Relative L2 Error:   {np.mean(l2_err_traj):.2%}")
    logger.info(f"Trajectory Mean Momentum Residual:    {mean_pako:.6e}")
    logger.info("=" * 60)

    # Plotting Setup
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Model Evaluation: {model_path.name} | Relative L2 Error: {final_l2_error:.2%}",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: Training Convergence
    metrics_path = output_dir / f"metrics_{model_name}.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        axes[0, 0].plot(df["l_step"], "k-", label="Total Loss", linewidth=1.5)
        if "l_data" in df.columns:
            axes[0, 0].plot(df["l_data"], "r--", label="Data Loss", alpha=0.7)
        if "l_pako" in df.columns:
            axes[0, 0].plot(df["l_pako"], "b--", label="Momentum Loss", alpha=0.7)
        if "l_dirichlet" in df.columns:
            axes[0, 0].plot(df["l_dirichlet"], "g:", label="Dirichlet Loss", alpha=0.7)
        if "l_neumann" in df.columns:
            axes[0, 0].plot(df["l_neumann"], "m:", label="Neumann Loss", alpha=0.7)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Training Loss Convergence")
        axes[0, 0].set_xlabel("Logged Step")
        axes[0, 0].set_ylabel("Loss (Log Scale)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")

    # Subplot 2: Deformed Mesh State t_final
    x_nodes_np = mesh.nodes.cpu().numpy()
    u_final_np = all_u_pred[-1].numpy()
    u_mag = np.linalg.norm(u_final_np, axis=-1)

    sc = axes[0, 1].scatter(
        x_nodes_np[:, 0] + u_final_np[:, 0],
        x_nodes_np[:, 1] + u_final_np[:, 1],
        c=u_mag,
        cmap="viridis",
        s=40,
    )
    plt.colorbar(sc, ax=axes[0, 1], label=r"$\|\mathbf{u}\|$ Magnitude [m]")
    axes[0, 1].scatter(
        x_nodes_np[:, 0], x_nodes_np[:, 1], c="gray", alpha=0.3, s=15, label="Reference"
    )
    axes[0, 1].set_title("Final Deformed State ($t = T$)")
    axes[0, 1].set_xlabel("X [m]")
    axes[0, 1].set_ylabel("Y [m]")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Momentum Residual Norm
    res_free = residual_free.numpy()
    axes[0, 2].hist(
        res_free, bins=30, color="crimson", alpha=0.7, edgecolor="black"
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Momentum Balance Residual Norm")
    axes[0, 2].set_xlabel(r"$|R_{\mathrm{mom,free}}|$")
    axes[0, 2].set_ylabel("Count (Log Scale)")
    axes[0, 2].grid(True, alpha=0.3)

    # Subplot 4: Stress-Strain Response (S11 vs E11)
    axes[1, 0].plot(
        E11_gt_traj,
        S11_gt_traj,
        "k--",
        linewidth=2.0,
        label=r"Ground Truth / Theoretical",
    )
    axes[1, 0].plot(
        E11_pred_traj,
        S11_pred_traj,
        "r-o",
        linewidth=1.5,
        markersize=4,
        alpha=0.85,
        label=r"MeshGraphNet Prediction",
    )
    axes[1, 0].set_title("Constitutive Response ($S_{11}$ vs $E_{11}$)")
    axes[1, 0].set_xlabel(r"Green-Lagrange Strain $E_{11}$ [-]")
    axes[1, 0].set_ylabel(r"Second Piola-Kirchhoff Stress $S_{11}$ [Pa]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # Subplot 5: Tip Stress Trajectory (S11 vs Time)
    t_plot = time_grid.cpu().numpy()
    axes[1, 1].plot(
        t_plot, S11_gt_traj, "k--", linewidth=1.8, label=r"Ground Truth $S_{11}(X=L)$"
    )
    axes[1, 1].plot(
        t_plot, S11_pred_traj, "r-", linewidth=1.8, label=r"Predicted $S_{11}(X=L)$"
    )
    axes[1, 1].set_title("Tip Stress Trajectory ($X = L$)")
    axes[1, 1].set_xlabel("Time $t$ [s]")
    axes[1, 1].set_ylabel("Stress $S_{11}$ [Pa]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    # Subplot 6: Relative L2 Error over Time
    axes[1, 2].plot(
        t_plot,
        l2_err_traj,
        "g-",
        linewidth=1.8,
        label=r"Relative $L_2$ Displacement Error",
    )
    axes[1, 2].set_title("Displacement Error Trajectory")
    axes[1, 2].set_xlabel("Time $t$ [s]")
    axes[1, 2].set_ylabel("Relative $L_2$ Error")
    axes[1, 2].legend()
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"{model_path.stem}_evaluation.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved evaluation plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    if cfg.get("bulk", False):
        pt_files = [p for p in out_dir.glob("*.pt") if "checkpoints" not in str(p)]
        for model_file in pt_files:
            test(model_file)
    else:
        Tk().withdraw()
        selected_model = askopenfilename(initialdir=out_dir)
        if selected_model:
            test(Path(selected_model))
