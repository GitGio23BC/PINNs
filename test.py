import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.geometry import create_mesh
from src.loader import generate_ground_truth
from src.loss import rl2e
from src.models import BACKBONE_REGISTRY, ParametricPINN
from src.physics import (
    HUGO,
    HolzapfelEnergy_2D,
    haslach_constitutive_residual_2D,
    pako_residual_2D,
)
from src.utils import init_logging, load_config


def test(model_path: Path | None = None):
    # Logging and initialisation
    init_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config("config.yaml")
    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(cfg.get("output_dir", "./output"))

    if model_path is None:
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"
    model_name = model_path.stem

    # Domain and Mesh
    x_range = cfg["domain"]["x_range"]
    y_range = cfg["domain"]["y_range"]
    t_range = cfg["domain"]["t_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_range[-1] - x_range[0]),
        height=float(y_range[-1] - y_range[0]),
        nx=nx,
        ny=ny,
        device=device,
    )
    nodes = mesh.nodes

    # Ground Truth Dataset
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"].to(device)
    u_exact_traj = dataset["u"].to(device)
    trac_ext_traj = dataset["trac_ext"].to(device)
    time_steps = len(time_grid)

    # Instantiate Model and Load Checkpoint
    arch_cls = BACKBONE_REGISTRY[cfg["model"]["architecture"]]
    backbone = arch_cls(
        in_dim=3,
        hidden_layers=int(cfg["model"]["num_layers"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        out_dim=5,
    )
    pinn = ParametricPINN(
        backbone=backbone,
        x_range=x_range,
        y_range=y_range,
        t_range=t_range,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    pinn.load_state_dict(state_dict)
    pinn.eval()
    logger.info(f"Loaded trained PINN model from {model_path}")

    # Physics Constants and Constitutive Model
    p_cfg = cfg["physics"]
    visco_model = HUGO(
        HolzapfelEnergy_2D(
            c=float(p_cfg["c"]),
            c1=float(p_cfg["c1"]),
            c2=float(p_cfg["c2"]),
            c3=float(p_cfg["c3"]),
            device=device,
        ),
        k_relax=float(p_cfg["k_relax"]),
    )
    b = torch.tensor(p_cfg["body_force"], device=device, dtype=torch.float32)

    # Evaluation Trajectory Containers
    all_u_pred = []
    all_S_pred = []
    all_haslach_res = []
    all_pako_res = []

    logger.info("Starting PINN continuous trajectory evaluation...")

    for t_step in range(time_steps):
        t_val = time_grid[t_step].item()
        num_nodes = nodes.shape[0]

        # Space-time collocation tensors with autograd enabled for continuous PDE residuals
        t_in = torch.full((num_nodes, 1), t_val, device=device, requires_grad=True)
        X_in = nodes.clone().detach().requires_grad_(True)

        predictions = pinn(t=t_in, X=X_in)
        u_pred = predictions[:, :2]
        S_pred = predictions[:, 2:]

        # Constitutive and Momentum Balance Residuals
        haslach_res = haslach_constitutive_residual_2D(
            u_pred=u_pred,
            S_pred=S_pred,
            X_ref=X_in,
            t=t_in,
            visco_model=visco_model,
        )

        pako_res = pako_residual_2D(
            u_pred=u_pred,
            S_pred=S_pred,
            X_ref=X_in,
            b=b,
        )

        all_u_pred.append(u_pred.detach().cpu())
        all_S_pred.append(S_pred.detach().cpu())
        all_haslach_res.append(haslach_res.detach().cpu())
        all_pako_res.append(pako_res.detach().cpu())

    # Metrics computation
    u_pred_traj = torch.stack(all_u_pred, dim=0)  # Shape: (T, N, 2)
    u_exact_cpu = u_exact_traj.cpu()

    # Step-wise relative L2 errors
    step_l2_errors = [
        rl2e(u_pred_traj[step], u_exact_cpu[step]).item()
        for step in range(time_steps)
    ]
    final_l2_error = step_l2_errors[-1]
    mean_trajectory_l2 = float(np.mean(step_l2_errors))

    mean_haslach = torch.cat(all_haslach_res).abs().mean().item()
    mean_pako = torch.cat(all_pako_res).abs().mean().item()

    logger.info("=" * 60)
    logger.info(f"Evaluation Results for: {model_path.name}")
    logger.info(f"Final Relative L2 Displacement Error:    {final_l2_error:.2%}")
    logger.info(f"Mean Trajectory Relative L2 Error:       {mean_trajectory_l2:.2%}")
    logger.info(f"Trajectory Mean Haslach Residual:        {mean_haslach:.6e}")
    logger.info(f"Trajectory Mean Momentum Residual:       {mean_pako:.6e}")
    logger.info("=" * 60)

    # Trajectory Boundary Condition Data Extraction
    t_plot = time_grid.cpu().numpy()
    left_nodes_idx = mesh.left_nodes.cpu()
    right_nodes_idx = mesh.right_nodes.cpu()

    base_u_norm_traj = [
        torch.linalg.norm(u[left_nodes_idx], dim=-1).mean().item()
        for u in all_u_pred
    ]
    tip_S11_pred_traj = [S[right_nodes_idx, 0].mean().item() for S in all_S_pred]
    tip_S11_target_traj = [
        trac_ext_traj[t, right_nodes_idx, 0].mean().item()
        for t in range(time_steps)
    ]

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"PINN Evaluation: {model_path.name} | Final L2 Error: {final_l2_error:.2%}",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: Training Convergence
    metrics_path = output_dir / f"metrics_{model_name}.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        axes[0, 0].plot(df["step_loss"], "k-", label="Total Loss", linewidth=1.5)
        if "loss_data" in df.columns:
            axes[0, 0].plot(df["loss_data"], "c-.", label="Data Loss", alpha=0.7)
        if "loss_haslach" in df.columns:
            axes[0, 0].plot(df["loss_haslach"], "r--", label="Constitutive Loss", alpha=0.7)
        if "loss_pako" in df.columns:
            axes[0, 0].plot(df["loss_pako"], "b--", label="Momentum Loss", alpha=0.7)
        if "loss_bc_base" in df.columns:
            axes[0, 0].plot(df["loss_bc_base"], "g:", label="BC Base Loss", alpha=0.7)
        if "loss_bc_tip" in df.columns:
            axes[0, 0].plot(df["loss_bc_tip"], "m:", label="BC Traction Loss", alpha=0.7)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Training Loss Convergence")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss (Log Scale)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "metrics.csv not found", ha="center", va="center")

    # Subplot 2: Final Deformed State at t_final
    x_nodes_np = mesh.nodes.cpu().numpy()
    u_final_pred = all_u_pred[-1].numpy()
    u_final_exact = u_exact_cpu[-1].numpy()
    u_mag = np.linalg.norm(u_final_pred, axis=-1)

    sc = axes[0, 1].scatter(
        x_nodes_np[:, 0] + u_final_pred[:, 0],
        x_nodes_np[:, 1] + u_final_pred[:, 1],
        c=u_mag,
        cmap="viridis",
        s=40,
        label="PINN Deformed",
    )
    plt.colorbar(sc, ax=axes[0, 1], label=r"$\|\mathbf{u}\|$ [m]")
    axes[0, 1].scatter(
        x_nodes_np[:, 0] + u_final_exact[:, 0],
        x_nodes_np[:, 1] + u_final_exact[:, 1],
        facecolors="none",
        edgecolors="red",
        s=40,
        linestyle="--",
        label="Ground Truth",
    )
    axes[0, 1].scatter(
        x_nodes_np[:, 0], x_nodes_np[:, 1], c="gray", alpha=0.3, s=15, label="Reference"
    )
    axes[0, 1].set_title("Final Deformed State ($t = T$)")
    axes[0, 1].set_xlabel("X [m]")
    axes[0, 1].set_ylabel("Y [m]")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Momentum Balance Residual Norm Distribution
    res_norm = torch.linalg.norm(torch.cat(all_pako_res), dim=-1).numpy()
    axes[0, 2].hist(res_norm, bins=30, color="crimson", alpha=0.7, edgecolor="black")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Momentum Residual Distribution")
    axes[0, 2].set_xlabel(r"$\|\mathbf{R}_{\mathrm{mom}}\|$")
    axes[0, 2].set_ylabel("Count (Log Scale)")
    axes[0, 2].grid(True, alpha=0.3)

    # Subplot 4: Base Clamped Condition Enforcement across Time
    axes[1, 0].plot(
        t_plot,
        base_u_norm_traj,
        "b-",
        linewidth=1.8,
        label=r"Predicted $\|\mathbf{u}(X=0)\|$",
    )
    axes[1, 0].axhline(
        0.0, color="k", linestyle="--", alpha=0.7, label="Exact Target (0.0)"
    )
    axes[1, 0].set_title("Clamped Base Boundary ($X = 0$)")
    axes[1, 0].set_xlabel("Time $t$ [s]")
    axes[1, 0].set_ylabel("Displacement Norm [m]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # Subplot 5: Traction Enforcement at Tip across Time
    axes[1, 1].plot(
        t_plot, tip_S11_pred_traj, "r-", linewidth=1.8, label=r"Predicted $S_{11}(X=1)$"
    )
    axes[1, 1].plot(
        t_plot,
        tip_S11_target_traj,
        "k--",
        linewidth=1.5,
        label=r"Target Traction $S_{11}$",
    )
    axes[1, 1].set_title("Tip Stress Tracking ($X = 1$)")
    axes[1, 1].set_xlabel("Time $t$ [s]")
    axes[1, 1].set_ylabel("Stress $S_{11}$ [Pa]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    # Subplot 6: Constitutive Haslach Residual Norm Distribution
    haslach_norm = torch.linalg.norm(torch.cat(all_haslach_res), dim=-1).numpy()
    axes[1, 2].hist(
        haslach_norm, bins=30, color="darkorange", alpha=0.7, edgecolor="black"
    )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Haslach Constitutive Residual Distribution")
    axes[1, 2].set_xlabel(r"$\|\mathbf{R}_{\mathrm{visco}}\|$")
    axes[1, 2].set_ylabel("Count (Log Scale)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"{model_path.stem}_evaluation.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved evaluation plot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    cfg = load_config("config.yaml")
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