from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.geometry import create_mesh_3D
from src.loader import generate_inertial_dataset
from src.utils import init_logging, load_config


def animate_ground_truth(
    data_name: str|None = None,
    fps: int = 20,
):
    init_logging()
    cfg = load_config()
    device = torch.device("cpu")
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset loading / generation
    data_dir = Path(cfg["data"]["data_dir"])
    if data_name is None:
        data_name = str(cfg["data"]["dataset_name"])
    
    dataset_path = data_dir / data_name
    output_filename = data_name.split(".pt")[0]+".gif"

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Generating now...")
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        print(f"Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"].cpu().numpy()
    u_exact_traj = dataset["u"].cpu().numpy()
    num_frames = len(time_grid)

    # Mesh creation
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    z_min, z_max = cfg["domain"]["z_range"]
    nx, ny, nz = (
        int(cfg["domain"]["nx"]),
        int(cfg["domain"]["ny"]),
        int(cfg["domain"]["nz"]),
    )

    mesh = create_mesh_3D(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        depth=float(z_max - z_min),
        nx=nx,
        ny=ny,
        nz=nz,
        device=device,
    )

    nodes_ref = mesh.nodes.cpu().numpy()
    if u_exact_traj.shape[1:] != nodes_ref.shape:
        raise ValueError(
            "Dataset displacement shape does not match the configured 3D mesh: "
            f"{u_exact_traj.shape[1:]} != {nodes_ref.shape}."
        )

    boundary_faces = torch.cat(list(mesh.boundary_faces.values()), dim=0).cpu().numpy()
    right_nodes = mesh.boundary_nodes["right"].cpu().numpy()
    traction_traj = dataset["trac_ext"].cpu().numpy()
    if traction_traj.ndim == 2:
        traction_traj = np.broadcast_to(
            traction_traj[:, None, :],
            (len(time_grid), len(right_nodes), 3),
        )
    if traction_traj.shape != (len(time_grid), len(right_nodes), 3):
        raise ValueError(
            "Dataset traction shape does not match the configured right boundary: "
            f"{traction_traj.shape} != {(len(time_grid), len(right_nodes), 3)}."
        )

    mag_exact = np.linalg.norm(u_exact_traj, axis=-1)
    vmax = max(float(mag_exact.max()), 1e-4)
    color_norm = plt.Normalize(vmin=0.0, vmax=vmax) # type: ignore
    positions = nodes_ref[None, :, :] + u_exact_traj
    bounds_min = positions.reshape(-1, 3).min(axis=0)
    bounds_max = positions.reshape(-1, 3).max(axis=0)
    spans = bounds_max - bounds_min
    margin = np.maximum(spans * 0.1, max(mesh.width, mesh.height, mesh.depth) * 0.02)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scalar_map = plt.cm.ScalarMappable(cmap="viridis", norm=color_norm)
    fig.colorbar(
        scalar_map,
        ax=ax,
        shrink=0.7,
        pad=0.1,
        label="Displacement Magnitude ||u|| [m]",
    )

    def update(frame: int):
        ax.clear()
        pos_current = positions[frame]
        face_magnitude = mag_exact[frame][boundary_faces].mean(axis=1)
        surface = Poly3DCollection(
            pos_current[boundary_faces],
            facecolors=plt.cm.viridis(color_norm(face_magnitude)),
            edgecolors=(0.1, 0.1, 0.1, 0.35),
            linewidths=0.25,
        )
        ax.add_collection3d(surface)

        traction_current = traction_traj[frame]
        traction_norm = np.linalg.norm(traction_current, axis=-1)
        max_traction = float(traction_norm.max()) if traction_norm.size else 0.0
        active_nodes = traction_norm > max(max_traction * 1e-6, 1e-8)
        if np.any(active_nodes):
            arrow_length = 0.18 * max(mesh.width, mesh.height, mesh.depth)
            arrows = traction_current[active_nodes] * (arrow_length / max_traction)
            arrow_origins = pos_current[right_nodes[active_nodes]]
            ax.quiver(
                arrow_origins[:, 0],
                arrow_origins[:, 1],
                arrow_origins[:, 2],
                arrows[:, 0],
                arrows[:, 1],
                arrows[:, 2],
                color="red",
                arrow_length_ratio=0.2,
            )
            ax.text2D(
                0.02,
                0.96,
                "Red arrows: applied traction (scaled)",
                transform=ax.transAxes,
            )

        ax.set_title(
            f"Dataset Ground Truth Verification (t = {float(time_grid[frame]):.3f} s)"
        )
        ax.set_xlim(bounds_min[0] - margin[0], bounds_max[0] + margin[0])
        ax.set_ylim(bounds_min[1] - margin[1], bounds_max[1] + margin[1])
        ax.set_zlim(bounds_min[2] - margin[2], bounds_max[2] + margin[2])
        ax.set_box_aspect((mesh.width, mesh.height, mesh.depth))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=25, azim=-55)
        ax.grid(True, linestyle="--", alpha=0.3)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000 // fps, blit=False
    )

    save_target = output_dir / output_filename

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
    plt.close(fig)
    print(f"Animation successfully exported to: {save_target}")


if __name__ == "__main__":
    animate_ground_truth()
