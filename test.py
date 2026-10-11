import logging
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


def test(model_path: Path | None = None):
    # Logging and set-up
    init_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config()
    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(cfg.get("output_dir", "./output"))

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

    data_loader = MGNData(dataset, mesh)
    time_steps = len(dataset["time"])
    dt = data_loader.dt
    time_grid = dataset["time"]

    # Model
    mgn = MeshGraphNet().to(device=device, dtype=torch.float64)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mgn.load_state_dict(state_dict)
    mgn.eval()
    logger.info(f"Loaded trained MeshGraphNet model from {model_path}")

    # Physics Constants
    p_cfg = cfg["physics"]
    tissue = p_cfg["tissue"]
    alphas = torch.tensor(p_cfg["alphas"][tissue], device=device, dtype=torch.float64)
    mus_list = [float(mu) for mu in p_cfg["mus"][tissue]]
    mus = torch.tensor(mus_list, device=device, dtype=torch.float64)
    lam = float(p_cfg["lam"])
    beta = float(p_cfg["beta"])
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float64)
    rho = float(p_cfg["rho_0"])

    # Rollout Initialisation
    u_prev = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64, device=device)
    u_prev2 = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64, device=device)
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

        graph = create_graph(
            mesh=mesh,
            world_pos=world_pos,
            node_type=node_type,
            device=device,
        )

        with torch.enable_grad():
            u = mgn(graph).to(dtype=torch.float64)
            u.requires_grad_(True)

            # Acceleration & External Force
            u_tt = (u - 2.0 * u_prev + u_prev2) / (dt**2)
            F_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64, device=device)
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
            u_exact = batch.u_exact.to(device=device).to(torch.float64).detach().clone()
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
        all_pako_res.append(R_pako.detach().cpu())

        u_prev2 = u_prev.detach().clone()
        u_prev = u.detach().clone()
        world_pos = (mesh.nodes + u_prev).detach()

    # Metrics computation
    l2_err_traj = [
        rl2e(u_pred, dataset["u"][t].cpu()).item()
        for t, u_pred in enumerate(all_u_pred)
    ]
    final_l2_error = l2_err_traj[-1]
    mean_pako = torch.cat(all_pako_res).abs().mean().item()

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
    res_norm = torch.linalg.norm(torch.cat(all_pako_res, dim=0), dim=-1).numpy()
    axes[0, 2].hist(res_norm, bins=30, color="crimson", alpha=0.7, edgecolor="black")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Momentum Balance Residual Norm")
    axes[0, 2].set_xlabel(r"$\|\mathbf{R}_{\mathrm{mom}}\|$")
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
