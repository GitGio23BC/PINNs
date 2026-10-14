from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, PillowWriter

from src.geometry import create_graph, create_mesh
from src.loader import generate_inertial_dataset
from src.models import MeshGraphNet
from src.utils import init_logging, load_config


def rollout_trajectory(
    model: MeshGraphNet,
    mesh,
    node_type: torch.Tensor,
    time_steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    u_prev = torch.zeros((mesh.n_nodes, 2), dtype=torch.float32, device=device)
    world_pos = mesh.nodes.clone()

    u_pred_trajectory = []

    with torch.no_grad():
        for _ in range(time_steps):
            graph = create_graph(
                mesh=mesh,
                world_pos=world_pos,
                node_type=node_type,
                device=device,
            )
            u = model(graph).to(dtype=torch.float32)
            u_pred_trajectory.append(u.cpu())

            u_prev = u.detach().clone()
            world_pos = (mesh.nodes + u_prev).detach()

    return torch.stack(u_pred_trajectory, dim=0).numpy()


def animate_deformation(
    model_path: Path | None = None,
    output_filename: str = "mesh_deformation.mp4",
    fps: int = 20,
):
    # Config and logging
    init_logging()
    cfg = load_config()
    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

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
        print(f"Dataset not found at {dataset_path}. Generating now...")
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"].cpu().numpy()
    u_exact_traj = dataset["u"].cpu().numpy()
    num_frames = len(time_grid)

    F_ext_traj = np.zeros((num_frames, mesh.n_nodes, 2), dtype=np.float32)

    trac_traj = dataset["trac_ext"].cpu().numpy()
    right_nodes = mesh.right_nodes.cpu().numpy()
    right_lengths = mesh.right_lengths.cpu().numpy()

    if trac_traj.ndim == 3:
        F_ext_traj[:, right_nodes, :] = trac_traj * right_lengths[None, :, None]
    elif trac_traj.ndim == 2:
        F_ext_traj[:, right_nodes, :] = (
            trac_traj[:, None, :] * right_lengths[None, :, None]
        )

    # Model 
    mgn = MeshGraphNet().to(device=device, dtype=torch.float32)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mgn.load_state_dict(state_dict)

    # Rollout predicted trajectory
    u_pred_traj = rollout_trajectory(mgn, mesh, node_type, num_frames, device)

    # Geometry bounds & plot setup
    nodes_ref = mesh.nodes.cpu().numpy()
    elements = mesh.elements.cpu().numpy()

    all_x = np.concatenate(
        [nodes_ref[:, 0] + u_exact_traj[..., 0], nodes_ref[:, 0] + u_pred_traj[..., 0]]
    )
    all_y = np.concatenate(
        [nodes_ref[:, 1] + u_exact_traj[..., 1], nodes_ref[:, 1] + u_pred_traj[..., 1]]
    )
    margin_x = (all_x.max() - all_x.min()) * 0.15
    margin_y = (all_y.max() - all_y.min()) * 0.15

    xlim = (all_x.min() - margin_x, all_x.max() + margin_x)
    ylim = (all_y.min() - margin_y, all_y.max() + margin_y)

    mag_exact = np.linalg.norm(u_exact_traj, axis=-1)
    mag_pred = np.linalg.norm(u_pred_traj, axis=-1)
    vmax = max(mag_exact.max(), mag_pred.max(), 1e-4)

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=vmax))  # type: ignore
    fig.colorbar(
        sm,
        ax=[ax_gt, ax_pred],
        orientation="horizontal",
        fraction=0.05,
        pad=0.15,
        label="Displacement Magnitude ||u|| [m]",
    )

    def update(frame: int):
        t_val = float(time_grid[frame])

        f_current = F_ext_traj[frame]
        f_norm = np.linalg.norm(f_current, axis=-1)
        active_nodes = f_norm > 1e-8

        # Ground truth plot
        pos_gt = nodes_ref + u_exact_traj[frame]
        ax_gt.clear()
        ax_gt.set_title(f"Ground Truth (t = {t_val:.3f} s)")
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

        if np.any(active_nodes):
            ax_gt.quiver(
                pos_gt[active_nodes, 0],
                pos_gt[active_nodes, 1],
                f_current[active_nodes, 0],
                f_current[active_nodes, 1],
                color="red",
                scale=None,
                width=0.005,
                label=r"Applied $F_{ext}$",
            )
            ax_gt.legend(loc="upper left")

        ax_gt.set_xlim(xlim)
        ax_gt.set_ylim(ylim)
        ax_gt.set_aspect("equal")
        ax_gt.grid(True, linestyle="--", alpha=0.3)
        ax_gt.set_xlabel("X [m]")
        ax_gt.set_ylabel("Y [m]")

        # Prediction plot
        pos_pred = nodes_ref + u_pred_traj[frame]
        ax_pred.clear()
        ax_pred.set_title(f"MeshGraphNet PINN (t = {t_val:.3f} s)")
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

        if np.any(active_nodes):
            ax_pred.quiver(
                pos_pred[active_nodes, 0],
                pos_pred[active_nodes, 1],
                f_current[active_nodes, 0],
                f_current[active_nodes, 1],
                color="red",
                scale=None,
                width=0.005,
                label=r"Applied $F_{ext}$",
            )
            ax_pred.legend(loc="upper left")

        ax_pred.set_xlim(xlim)
        ax_pred.set_ylim(ylim)
        ax_pred.set_aspect("equal")
        ax_pred.grid(True, linestyle="--", alpha=0.3)
        ax_pred.set_xlabel("X [m]")

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000 // fps, blit=False
    )

    # Writer selection & execution
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps)
        save_target = output_dir / output_filename
    else:
        print(
            "FFmpeg not found on system. Falling back to PillowWriter (.gif output)..."
        )
        writer = PillowWriter(fps=fps)
        save_target = output_dir / f"{Path(output_filename).stem}.gif"

    anim.save(save_target, writer=writer)
    plt.close()
    print(f"Animation successfully exported to: {save_target}")


if __name__ == "__main__":
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    if cfg.get("bulk", False):
        pt_files = [p for p in out_dir.glob("*.pt") if "checkpoints" not in str(p)]
        for model_file in pt_files:
            animate_deformation(model_file, f"{model_file.stem}_mesh_deformation.mp4")
    else:
        Tk().withdraw()
        selected_model = askopenfilename(initialdir=out_dir)
        if selected_model:
            selected_model = Path(selected_model)
            animate_deformation(
                model_path=selected_model,
                output_filename=f"{selected_model.stem}_mesh_deformation.mp4",
            )
