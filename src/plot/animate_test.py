import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.geometry import create_graph, create_mesh_3D
from src.loader import generate_inertial_dataset
from src.models import MeshGraphNet
from src.utils import init_logging, load_config


def rollout_trajectory(
    model: MeshGraphNet,
    mesh,
    node_type: torch.Tensor,
    time_grid: np.ndarray,
    traction_trajectory: torch.Tensor,
    traction_scale: float,
    time_steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    u_prev = torch.zeros((mesh.n_nodes, 3), dtype=torch.float32, device=device)
    u_prev2 = torch.zeros_like(u_prev)
    dt = float((time_grid[-1] - time_grid[0]) / max(time_steps - 1, 1))
    velocity_prev = torch.zeros_like(u_prev)
    world_pos = mesh.nodes.clone()

    u_pred_trajectory = []

    with torch.no_grad():
        for t_n in range(time_steps):
            trac_feat = torch.zeros((mesh.n_nodes, 3), device=device, dtype=torch.float32)
            trac_feat[mesh.get_boundary_nodes("right")] = traction_trajectory[t_n] / traction_scale
            
            graph = create_graph(
                mesh=mesh,
                world_pos=world_pos,
                node_type=node_type,
                velocity=velocity_prev,
                boundary_traction=trac_feat,
                device=device,
                dtype=torch.float32
            )
            u = model(graph).to(dtype=torch.float32)
            u_pred_trajectory.append(u.cpu())

            u_prev2 = u_prev.detach().clone()
            u_prev = u.detach().clone()
            velocity_prev = ((u_prev - u_prev2) / dt).detach()
            world_pos = (mesh.nodes + u_prev).detach()

    return torch.stack(u_pred_trajectory, dim=0).numpy()


def animate_deformation(
    model_path: Path | None = None,
    output_filename: str | None = None,
    fps: int = 20,
):  
    dtype = torch.float32
    # Config and logging
    init_logging()
    cfg = load_config()
    device = torch.device(cfg["training"].get("device", "cpu"))
    output_dir = Path(cfg.get("output_dir", "./output")) / cfg['model']['model_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_filename is None:
        output_filename = cfg['model']['model_name'] + "_deformation.mp4"

    if model_path is None:
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    # Mesh 3D
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    z_min, z_max = cfg["domain"]["z_range"]
    nx, ny, nz = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"]), int(cfg["domain"]["nz"])

    mesh = create_mesh_3D(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        depth=float(z_max - z_min),
        nx=nx,
        ny=ny,
        nz=nz,
        device=device,
        dtype=dtype
    )

    # Allineamento ai bounday nodes a 6 facce
    directions = ('left', 'right', 'bottom', 'top', 'back', 'front')
    node_type = torch.zeros((mesh.n_nodes, len(directions)), dtype=dtype, device=device)
    for i, direction in enumerate(directions):
        node_type[mesh.get_boundary_nodes(direction), i] = 1.0

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Generating now...")
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

    time_grid = dataset["time"].cpu().numpy()
    u_exact_traj = dataset["u"].cpu().numpy()
    num_frames = len(time_grid)

    right_nodes = mesh.get_boundary_nodes("right").cpu().numpy()
    traction_traj = dataset["trac_ext"].cpu().numpy()
    
    # Broadcast della trazione come in animate_ground_truth
    if traction_traj.ndim == 2:
        traction_traj = np.broadcast_to(
            traction_traj[:, None, :],
            (len(time_grid), len(right_nodes), 3),
        )

    # Model
    model_cfg = cfg["model"]
    mgn = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=torch.float32)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mgn.load_state_dict(state_dict)

    # Rollout predicted trajectory
    u_pred_traj = rollout_trajectory(
        mgn,
        mesh,
        node_type,
        time_grid,
        dataset["trac_ext"],
        float(cfg["physics"]["sigma_max"]),
        num_frames,
        device,
    )

    # Rendering Setup
    nodes_ref = mesh.nodes.cpu().numpy()
    boundary_faces = torch.cat(list(mesh.boundary_faces.values()), dim=0).cpu().numpy()

    # Pre-calcola bounds per entrambe le traiettorie
    pos_exact = nodes_ref[None, :, :] + u_exact_traj
    pos_pred = nodes_ref[None, :, :] + u_pred_traj
    
    all_pos = np.concatenate([pos_exact, pos_pred], axis=0)
    bounds_min = all_pos.reshape(-1, 3).min(axis=0)
    bounds_max = all_pos.reshape(-1, 3).max(axis=0)
    spans = bounds_max - bounds_min
    margin = np.maximum(spans * 0.1, max(mesh.width, mesh.height, mesh.depth) * 0.02)

    mag_exact = np.linalg.norm(u_exact_traj, axis=-1)
    mag_pred = np.linalg.norm(u_pred_traj, axis=-1)
    vmax = max(float(mag_exact.max()), float(mag_pred.max()), 1e-4)
    color_norm = plt.Normalize(vmin=0.0, vmax=vmax)

    # Figure Layout
    fig = plt.figure(figsize=(18, 8))
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")
    
    scalar_map = plt.cm.ScalarMappable(cmap="viridis", norm=color_norm)
    fig.colorbar(
        scalar_map,
        ax=[ax_gt, ax_pred],
        shrink=0.7,
        pad=0.05,
        orientation="horizontal",
        fraction=0.04,
        label="Displacement Magnitude ||u|| [m]",
    )

    def update(frame: int):
        t_val = float(time_grid[frame])

        # === 1. GROUND TRUTH ===
        ax_gt.clear()
        pos_gt_current = pos_exact[frame]
        face_mag_gt = mag_exact[frame][boundary_faces].mean(axis=1)
        
        surf_gt = Poly3DCollection(
            pos_gt_current[boundary_faces],
            facecolors=plt.cm.viridis(color_norm(face_mag_gt)),
            edgecolors=(0.1, 0.1, 0.1, 0.35),
            linewidths=0.25,
        )
        ax_gt.add_collection3d(surf_gt)

        # Vector field (Traction) su GT
        traction_current = traction_traj[frame]
        traction_norm = np.linalg.norm(traction_current, axis=-1)
        max_traction = float(traction_norm.max()) if traction_norm.size else 0.0
        active_nodes = traction_norm > max(max_traction * 1e-6, 1e-8)
        
        if np.any(active_nodes):
            arrow_length = 0.18 * max(mesh.width, mesh.height, mesh.depth)
            arrows = traction_current[active_nodes] * (arrow_length / max_traction)
            arrow_origins = pos_gt_current[right_nodes[active_nodes]]
            ax_gt.quiver(
                arrow_origins[:, 0], arrow_origins[:, 1], arrow_origins[:, 2],
                arrows[:, 0], arrows[:, 1], arrows[:, 2],
                color="red", arrow_length_ratio=0.2,
            )

        ax_gt.set_title(f"Ground Truth (t = {t_val:.3f} s)")

        # === 2. MESH GRAPH NET PREDICTION ===
        ax_pred.clear()
        pos_pred_current = pos_pred[frame]
        face_mag_pred = mag_pred[frame][boundary_faces].mean(axis=1)

        surf_pred = Poly3DCollection(
            pos_pred_current[boundary_faces],
            facecolors=plt.cm.viridis(color_norm(face_mag_pred)),
            edgecolors=(0.1, 0.1, 0.1, 0.35),
            linewidths=0.25,
        )
        ax_pred.add_collection3d(surf_pred)

        # Vector field (Traction) su Prediction
        if np.any(active_nodes):
            arrow_origins_pred = pos_pred_current[right_nodes[active_nodes]]
            ax_pred.quiver(
                arrow_origins_pred[:, 0], arrow_origins_pred[:, 1], arrow_origins_pred[:, 2],
                arrows[:, 0], arrows[:, 1], arrows[:, 2],
                color="red", arrow_length_ratio=0.2,
            )
            ax_pred.text2D(0.02, 0.96, "Red arrows: applied traction", transform=ax_pred.transAxes)

        ax_pred.set_title(f"MeshGraphNet PINN (t = {t_val:.3f} s)")

        # === IMPOSTAZIONI COMUNI (Da rieseguire dopo ogni .clear()) ===
        for ax in (ax_gt, ax_pred):
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

    # Writer selection & execution
    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps)
        save_target = output_dir / output_filename
    else:
        print("FFmpeg not found on system. Falling back to PillowWriter (.gif output)...")
        writer = PillowWriter(fps=fps)
        save_target = output_dir / f"{Path(output_filename).stem}.gif"

    anim.save(save_target, writer=writer)
    plt.close(fig)
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