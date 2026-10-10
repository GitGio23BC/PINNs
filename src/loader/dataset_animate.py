from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from src.geometry import create_mesh
from src.loader import generate_ground_truth
from src.utils import init_logging, load_config


def animate_ground_truth(
    output_filename: str = "dataset_verification.mp4",
    fps: int = 20,
):
    init_logging()
    cfg = load_config()
    device = torch.device("cpu")
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Generating now...")
        dataset = generate_ground_truth(cfg, device=device)
    else:
        print(f"Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"].cpu().numpy()
    u_exact_traj = dataset["u"].cpu().numpy()        
    F_ext_traj = dataset["F_ext"].cpu().numpy()    

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

    nodes_ref = mesh.nodes.cpu().numpy()
    elements = mesh.elements.cpu().numpy()
    num_frames = len(time_grid)

    # Get max displacement and margin
    mag_exact = np.linalg.norm(u_exact_traj, axis=-1)
    vmax = max(mag_exact.max(), 1e-4)

    all_x = nodes_ref[:, 0] + u_exact_traj[..., 0]
    all_y = nodes_ref[:, 1] + u_exact_traj[..., 1]
    margin_x = (all_x.max() - all_x.min()) * 0.15
    margin_y = (all_y.max() - all_y.min()) * 0.15

    xlim = (all_x.min() - margin_x, all_x.max() + margin_x)
    ylim = (all_y.min() - margin_y, all_y.max() + margin_y)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=vmax) # type: ignore
    )
    fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.12,
        label="Displacement Magnitude ||u|| [m]",
    )

    def update(frame: int):
        ax.clear()
        t_val = float(time_grid[frame])
        pos_current = nodes_ref + u_exact_traj[frame]

        ax.set_title(f"Dataset Ground Truth Verification (t = {t_val:.3f} s)")

        # Colored mesh deformation
        ax.tripcolor(
            pos_current[:, 0],
            pos_current[:, 1],
            elements,
            mag_exact[frame],
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            shading="gouraud",
        ) # pyright: ignore[reportCallIssue]

        # Mesh plot
        ax.triplot(
            pos_current[:, 0],
            pos_current[:, 1],
            elements,
            color="black",
            alpha=0.3,
            linewidth=0.8,
        )

        # Plot F_ext
        f_current = F_ext_traj[frame]
        f_norm = np.linalg.norm(f_current, axis=-1)
        active_nodes = f_norm > 1e-8

        if np.any(active_nodes):
            ax.quiver(
                pos_current[active_nodes, 0],
                pos_current[active_nodes, 1],
                f_current[active_nodes, 0],
                f_current[active_nodes, 1],
                color="red",
                scale=None,
                width=0.005,
                label=r"Applied $F_{ext}$",
            )
            ax.legend(loc="upper left")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000 // fps, blit=False
    )

    save_target = output_dir / output_filename
    anim.save(save_target, writer="ffmpeg", fps=fps)
    plt.close()
    print(f"Dataset verification animation saved to: {save_target}")


if __name__ == "__main__":
    animate_ground_truth()