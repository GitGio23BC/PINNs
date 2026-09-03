import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.geometry import create_mesh, create_time_graph
from src.loader import generate_ground_truth
from src.loss import rl2e
from src.models import MeshGraphNet
from src.physics import (
    haslach_constitutive_residual_2D,
    pako_residual_2D,
)
from src.utils import init_logging, load_config


def test(model_path: Path | None = None):
    # Logging and set-up
    init_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config()
    device = "cpu"
    output_dir = Path(cfg.get("output_dir", "./output"))
    model_name = cfg["model"]["model_name"]


    if model_path is None:
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"
    model_name = model_path.stem

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"]
    t_min, t_max = time_grid[0].item(), time_grid[-1].item()
    time_steps = len(time_grid)
    u_exact_traj = dataset["u"]
    F_trajectory = dataset.get("F_applied", None)

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
    num_nodes = mesh.n_nodes
    boundary_indices = mesh.boundary_nodes

    tol = 1e-7
    base_mask = mesh.nodes[boundary_indices, 0] <= (x_min + tol)
    tip_mask = mesh.nodes[boundary_indices, 0] >= (x_max - tol)

    base_nodes_idx = boundary_indices[base_mask].cpu()
    tip_nodes_idx = boundary_indices[tip_mask].cpu()

    # Model
    mgn = MeshGraphNet(
        node_in_dim=int(cfg["model"]["node_in_dim"]),
        edge_in_dim=int(cfg["model"]["edge_in_dim"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        output_dim=int(cfg["model"].get("output_dim", 5)),
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mgn.load_state_dict(state_dict)
    mgn.eval()
    logger.info(f"Loaded trained MeshGraphNet model from {model_path}")

    # Physics Constants
    p_cfg = cfg["physics"]
    A = float(p_cfg.get("area", 1.0))
    k_relax = float(p_cfg["k_relax"])
    c_iso = float(p_cfg["c"])
    c1 = float(p_cfg["c1"])
    c2 = float(p_cfg["c2"])
    c3 = float(p_cfg["c3"])
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float32)

    time_steps = len(time_grid)
    dt = float((time_grid[-1] - time_grid[0]) / max(time_steps - 1, 1))

    # Initialisation
    E_prev_voigt = torch.zeros((num_nodes, 3), device=device, dtype=torch.float32)
    u_prev = torch.zeros((num_nodes, 2), device=device, dtype=torch.float32)

    all_u_pred = []
    all_S_pred = []
    all_haslach_res = []
    all_pako_res = []

    # Testing
    logger.info("Statirn PINN tetstseting...")
    for t_step in range(1, time_steps):

        t_curr = time_grid[t_step]
        t_norm = (t_curr - t_min) / (t_max - t_min)
        graph = create_time_graph(mesh=mesh, u=u_prev, t=t_norm, device=device)
        predictions = mgn(graph)

        X_ref = graph.mesh_nodes

        u_pred = predictions[:, :2]
        S_pred = predictions[:, 2:]

        # Viscoelastic Residuals
        E_curr_voigt, haslach_res = haslach_constitutive_residual_2D(
            u_pred=u_pred,
            S_pred=S_pred,
            X_ref=X_ref,
            E_prev=E_prev_voigt,
            dt=dt,
            k_relax=k_relax,
            c=c_iso,
            c1=c1,
            c2=c2,
            c3=c3,
        )

        # Momentum Residual
        pako_res = pako_residual_2D(
            u_pred=u_pred,
            S_pred=S_pred,
            X_ref=X_ref,
            b=b,
        )

        all_u_pred.append(u_pred.detach().cpu())
        all_S_pred.append(S_pred.detach().cpu())
        all_haslach_res.append(haslach_res.detach().cpu())
        all_pako_res.append(pako_res.detach().cpu())

        E_prev_voigt = E_curr_voigt.detach()
        u_prev = u_pred.detach()

    # Metrics computation
    u_final_pred = all_u_pred[-1]
    u_final_exact = u_exact_traj[-1].cpu()
    final_l2_error = rl2e(u_final_pred, u_final_exact).item()
    mean_haslach = torch.cat(all_haslach_res).abs().mean().item()
    mean_pako = torch.cat(all_pako_res).abs().mean().item()

    logger.info("=" * 60)
    logger.info(f"Evaluation Results for: {model_path.name}")
    logger.info(f"Final Relative L2 Displacement Error: {final_l2_error:.2%}")
    logger.info(f"Trajectory Mean Haslach Residual:     {mean_haslach:.6e}")
    logger.info(f"Trajectory Mean Momentum Residual:    {mean_pako:.6e}")
    logger.info("=" * 60)

    # Trajectory Boundary Condition Data Extraction
    t_plot = time_grid[1:].cpu().numpy()
    base_u_norm_traj = [
        torch.linalg.norm(u[base_nodes_idx], dim=-1).mean().item()
        for u in all_u_pred[1:]
    ]
    tip_S11_pred_traj = [S[tip_nodes_idx, 0].mean().item() for S in all_S_pred[1:]]

    if F_trajectory is not None:
        tip_S11_applied_traj = [
            (F_trajectory[t][tip_nodes_idx, 0] / A).mean().item()
            for t in range(1, time_steps)
        ]
    else:
        tip_S11_applied_traj = None

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Model Evaluation: {model_path.name} | Relative L2 Error: {final_l2_error:.2%}",
        fontsize=13,
        fontweight="bold",
    )

    # Plot Training Convergence
    metrics_path = output_dir / f"metrics_{model_name}.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        axes[0, 0].plot(df["step_loss"], "k-", label="Total Loss", linewidth=1.5)
        if "loss_haslach" in df.columns:
            axes[0, 0].plot(
                df["loss_haslach"], "r--", label="Constitutive Loss", alpha=0.7
            )
        if "loss_pako" in df.columns:
            axes[0, 0].plot(df["loss_pako"], "b--", label="Momentum Loss", alpha=0.7)
        if "loss_bc_base" in df.columns:
            axes[0, 0].plot(df["loss_bc_base"], "g:", label="BC Base Loss", alpha=0.7)
        if "loss_bc_tip" in df.columns:
            axes[0, 0].plot(df["loss_bc_tip"], "m:", label="BC Tip Loss", alpha=0.7)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Training Loss Convergence")
        axes[0, 0].set_xlabel("Logged Step")
        axes[0, 0].set_ylabel("Loss (Log Scale)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")

    # Plot Deformed Mesh State t_final
    x_nodes_np = mesh.nodes.cpu().numpy()
    u_final_np = u_final_pred.numpy()
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

    # Plot 3 Momentum Residual
    res_norm = torch.linalg.norm(torch.cat(all_pako_res), dim=-1).numpy()
    axes[0, 2].hist(res_norm, bins=30, color="crimson", alpha=0.7, edgecolor="black")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Momentum Balance Residual Norm")
    axes[0, 2].set_xlabel(r"$\|\mathbf{R}_{\mathrm{mom}}\|$")
    axes[0, 2].set_ylabel("Count (Log Scale)")
    axes[0, 2].grid(True, alpha=0.3)

    # Subplot 4 Boundary Condition at Base
    axes[1, 0].plot(
        t_plot,
        base_u_norm_traj,
        "b-",
        linewidth=1.8,
        label=r"Predicted $\|\mathbf{u}(X=0)\|$",
    )
    axes[1, 0].axhline(
        0.0, color="k", linestyle="--", alpha=0.7, label="Exact Dirichlet Target (0.0)"
    )
    axes[1, 0].set_title("Base Boundary Condition ($X = 0$)")
    axes[1, 0].set_xlabel("Time $t$ [s]")
    axes[1, 0].set_ylabel("Displacement Norm [m]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # Plot 5 Boundary Condition at Tip
    axes[1, 1].plot(
        t_plot, tip_S11_pred_traj, "r-", linewidth=1.8, label=r"Predicted $S_{11}(X=L)$"
    )
    if tip_S11_applied_traj is not None:
        axes[1, 1].plot(
            t_plot,
            tip_S11_applied_traj,
            "k--",
            linewidth=1.5,
            label=r"Applied Traction $S_{11}^{\mathrm{target}}$",
        )
    axes[1, 1].set_title("Tip Boundary Condition ($X = L$)")
    axes[1, 1].set_xlabel("Time $t$ [s]")
    axes[1, 1].set_ylabel("Stress $S_{11}$ [Pa]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    # Plot 6 Viscoelastic Residue
    haslach_norm = torch.linalg.norm(torch.cat(all_haslach_res), dim=-1).numpy()
    axes[1, 2].hist(
        haslach_norm, bins=30, color="darkorange", alpha=0.7, edgecolor="black"
    )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Haslach Constitutive Residual Norm")
    axes[1, 2].set_xlabel(r"$\|\mathbf{R}_{\mathrm{visco}}\|$")
    axes[1, 2].set_ylabel("Count (Log Scale)")
    axes[1, 2].grid(True, alpha=0.3)

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
