from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from src.geometry import create_mesh
from src.loader import generate_ground_truth
from src.models import BACKBONE_REGISTRY, ParametricPINN
from src.utils import init_logging, load_config


def evaluate_trajectory(
    model: ParametricPINN,
    nodes: torch.Tensor,
    time_grid: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    num_nodes = nodes.shape[0]
    time_steps = len(time_grid)

    u_pred_trajectory = []

    with torch.no_grad():
        for t_step in range(time_steps):
            t_val = time_grid[t_step].item()
            t_in = torch.full((num_nodes, 1), t_val, device=device)
            X_in = nodes.to(device)

            preds = model(t=t_in, X=X_in)
            u_pred = preds[:, :2].detach().cpu()
            u_pred_trajectory.append(u_pred)

    return torch.stack(u_pred_trajectory, dim=0).numpy()


def animate_deformation(
    model_path: Path | None = None,
    output_filename: str = "mesh_deformation.mp4",
    fps: int = 20,
):
    init_logging()
    cfg = load_config("config.yaml")
    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(cfg.get("output_dir", "./output"))

    if model_path is None:
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"]
    u_exact_traj = dataset["u"].cpu().numpy()

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

    # Model Instantiation and Checkpoint Loading
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

    # Continuous Trajectory Evaluation
    u_pred_traj = evaluate_trajectory(pinn, nodes, time_grid, device)

    # Initialise Animation
    nodes_ref = mesh.nodes.cpu().numpy()
    elements = mesh.elements.cpu().numpy()
    num_frames = len(time_grid)

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    all_x = np.concatenate(
        [nodes_ref[:, 0] + u_exact_traj[..., 0], nodes_ref[:, 0] + u_pred_traj[..., 0]]
    )
    all_y = np.concatenate(
        [nodes_ref[:, 1] + u_exact_traj[..., 1], nodes_ref[:, 1] + u_pred_traj[..., 1]]
    )
    margin_x = (all_x.max() - all_x.min()) * 0.1
    margin_y = (all_y.max() - all_y.min()) * 0.1

    xlim = (all_x.min() - margin_x, all_x.max() + margin_x)
    ylim = (all_y.min() - margin_y, all_y.max() + margin_y)

    mag_exact = np.linalg.norm(u_exact_traj, axis=-1)
    mag_pred = np.linalg.norm(u_pred_traj, axis=-1)
    vmax = max(float(mag_exact.max()), float(mag_pred.max()), 1e-4)

    # Initial plot elements
    ax_gt.tripcolor(
        nodes_ref[:, 0],
        nodes_ref[:, 1],
        elements,
        mag_exact[0],
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        shading="gouraud",
    )
    ax_gt.triplot(
        nodes_ref[:, 0],
        nodes_ref[:, 1],
        elements,
        color="black",
        alpha=0.3,
        linewidth=0.8,
    )

    tripcolor_pred = ax_pred.tripcolor(
        nodes_ref[:, 0],
        nodes_ref[:, 1],
        elements,
        mag_pred[0],
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        shading="gouraud",
    )
    ax_pred.triplot(
        nodes_ref[:, 0],
        nodes_ref[:, 1],
        elements,
        color="black",
        alpha=0.3,
        linewidth=0.8,
    )

    fig.colorbar(
        tripcolor_pred,
        ax=[ax_gt, ax_pred],
        orientation="horizontal",
        fraction=0.05,
        pad=0.15,
        label=r"Displacement Magnitude $\|\mathbf{u}\|$ [m]",
    )

    for ax in (ax_gt, ax_pred):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)

    ax_gt.set_xlabel("X [m]")
    ax_gt.set_ylabel("Y [m]")
    ax_pred.set_xlabel("X [m]")

    def update(frame: int):
        t_val = time_grid[frame].item()

        # Update Ground Truth View
        pos_gt = nodes_ref + u_exact_traj[frame]
        ax_gt.clear()
        ax_gt.set_title(f"Ground Truth ($t = {t_val:.3f}$ s)")
        ax_gt.tripcolor(
            pos_gt[:, 0],
            pos_gt[:, 1],
            elements,
            mag_exact[frame],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            shading="gouraud",
        )
        ax_gt.triplot(
            pos_gt[:, 0],
            pos_gt[:, 1],
            elements,
            color="black",
            alpha=0.3,
            linewidth=0.8,
        )
        ax_gt.set_xlim(xlim)
        ax_gt.set_ylim(ylim)
        ax_gt.set_aspect("equal")
        ax_gt.grid(True, linestyle="--", alpha=0.3)

        # Update PINN Prediction View
        pos_pred = nodes_ref + u_pred_traj[frame]
        ax_pred.clear()
        ax_pred.set_title(f"PINN Prediction ($t = {t_val:.3f}$ s)")
        ax_pred.tripcolor(
            pos_pred[:, 0],
            pos_pred[:, 1],
            elements,
            mag_pred[frame],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            shading="gouraud",
        )
        ax_pred.triplot(
            pos_pred[:, 0],
            pos_pred[:, 1],
            elements,
            color="black",
            alpha=0.3,
            linewidth=0.8,
        )
        ax_pred.set_xlim(xlim)
        ax_pred.set_ylim(ylim)
        ax_pred.set_aspect("equal")
        ax_pred.grid(True, linestyle="--", alpha=0.3)

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000 // fps, blit=False
    )

    save_target = output_dir / output_filename
    anim.save(save_target, writer="ffmpeg", fps=fps)

    plt.close()
    print(f"Animation successfully exported to: {save_target}")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    out_dir = Path(cfg["output_dir"])
    if cfg.get("bulk", False):
        pt_files = [p for p in out_dir.glob("*.pt") if "checkpoints" not in str(p)]
        for model_file in pt_files:
            animate_deformation(model_file, f"{model_file.stem}_mesh_deformation.mp4")
    else:
        Tk().withdraw()
        selected_model = askopenfilename(initialdir=out_dir)
        if selected_model:
            selected_path = Path(selected_model)
            animate_deformation(
                model_path=selected_path,
                output_filename=f"{selected_path.stem}_mesh_deformation.mp4",
            )