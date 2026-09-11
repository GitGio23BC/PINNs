import logging
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from src.geometry import create_graph, create_mesh
from src.loader import MGNData, generate_ground_truth
from src.models import FEATURE_DIMS, build_mgn_model
from src.utils import init_logging, load_config


def rollout_trajectory(
    model: torch.nn.Module,
    mesh,
    data_loader: MGNData,
    node_type: torch.Tensor,
    method: str,
    time_steps: int,
    time_grid: torch.Tensor,
    has_t: bool,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    num_nodes = mesh.n_nodes
    u_pred_trajectory = [np.zeros((num_nodes, 2), dtype=np.float32)]

    u_prev = torch.zeros((num_nodes, 2), device=device, dtype=torch.float32)
    u_dot_prev = torch.zeros((num_nodes, 2), device=device, dtype=torch.float32)
    E_prev_voigt = torch.zeros((num_nodes, 3), device=device, dtype=torch.float32)
    S_prev_voigt = torch.zeros((num_nodes, 3), device=device, dtype=torch.float32)

    with torch.no_grad():
        for t_step in range(1, time_steps):
            batch = data_loader.get_batch(t_step)
            t_norm = (
                (batch.t - time_grid[0]) / (time_grid[-1] - time_grid[0])
                if has_t
                else None
            )

            # Build graph conforming to the specified ablation tier
            if method == "base":
                graph = create_graph(
                    mesh=mesh, u=u_prev, node_type=node_type, t=t_norm, device=device
                )
            elif method == "traction":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    trac=batch.trac,
                    t=t_norm,
                    device=device,
                )
            elif method == "dynamic":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    u_dot=u_dot_prev,
                    trac=batch.trac,
                    t=t_norm,
                    device=device,
                )
            elif method == "visco":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    E_prev=E_prev_voigt,
                    trac=batch.trac,
                    t=t_norm,
                    device=device,
                )
            elif method == "full":
                graph = create_graph(
                    mesh=mesh,
                    u=u_prev,
                    node_type=node_type,
                    u_dot=u_dot_prev,
                    E_prev=E_prev_voigt,
                    S_prev=S_prev_voigt,
                    trac=batch.trac,
                    t=t_norm,
                    device=device,
                )
            else:
                raise KeyError(f"Evaluation method '{method}' not found")

            # Forward inference
            preds = model(graph)
            u_pred = preds[:, :2]
            S_pred = preds[:, 2:]

            u_dot_prev = (u_pred - u_prev) / batch.dt
            u_prev = u_pred
            S_prev_voigt = S_pred
            E_prev_voigt = batch.E

            u_pred_trajectory.append(u_pred.cpu().numpy())

    return np.stack(u_pred_trajectory, axis=0)


def animate_deformation(
    model_path: Path | None = None,
    method: str | None = None,
    output_filename: str | None = None,
    fps: int = 20,
):
    init_logging()
    logger = logging.getLogger("AnimateMGN")
    cfg = load_config("config.yaml")

    device = torch.device(cfg["training"].get("device", "cpu"))
    if model_path is None:
        output_dir = Path(cfg.get("output_dir", "./output"))
        model_path = output_dir / f"{cfg['model']['model_name']}.pt"
    else:
        output_dir = model_path.parent

    model_name = model_path.stem
    if output_filename is None:
        output_filename = f"{model_name}_mesh_deformation.mp4"

    # Infer ablation tier
    if method is None:
        method = model_name.split("_")[1]
        method = (
            method if method in FEATURE_DIMS else cfg["training"].get("method", "base")
        )

    # Load Checkpoint & Configuration
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    ckpt_cfg = checkpoint.get("config", {})
    if "type" in ckpt_cfg["model"]:
        model_type = ckpt_cfg["model"]["type"]
    else:
        model_type = cfg["model"].get("type", "standard")

    cfg["model"]["type"] = model_type

    # Dataset
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]
    if not dataset_path.exists():
        dataset = generate_ground_truth(cfg, device=device)
    else:
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"]
    time_steps = len(time_grid)
    data_loader = MGNData(cfg, dataset)
    u_exact_traj = dataset["u"].cpu().numpy()

    # Mesh
    mesh = create_mesh(
        width=float(cfg["domain"]["x_range"][1] - cfg["domain"]["x_range"][0]),
        height=float(cfg["domain"]["y_range"][1] - cfg["domain"]["y_range"][0]),
        nx=int(cfg["domain"]["nx"]),
        ny=int(cfg["domain"]["ny"]),
        device=device,
    )

    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)
    raw_node_type[mesh.top_nodes] = 3
    raw_node_type[mesh.bottom_nodes] = 3
    raw_node_type[mesh.right_nodes] = 2
    raw_node_type[mesh.left_nodes] = 1
    node_type = torch.nn.functional.one_hot(raw_node_type, num_classes=4).float()

    # Detect whether checkpoint incorporates temporal dimension t
    ckpt_dim = state_dict["encoder.node_encoder.input_layer.weight"].shape[1]
    expected_dim = FEATURE_DIMS[method]
    has_t = ckpt_dim == expected_dim

    # Initialise model and load weights
    mgn = build_mgn_model(cfg, method=method, device=device)
    mgn.load_state_dict(state_dict)
    logger.info(
        f"Loaded [{method.upper()}] | Type: {model_type} | Time Dim (t): {has_t}"
    )

    # Rollout Trajectory
    u_pred_traj = rollout_trajectory(
        model=mgn,
        mesh=mesh,
        data_loader=data_loader,
        node_type=node_type,
        method=method,
        time_steps=time_steps,
        time_grid=time_grid,
        has_t=has_t,
        device=device,
    )

    # Animation Setup
    nodes_ref = mesh.nodes.cpu().numpy()
    elements = mesh.elements.cpu().numpy()

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
        ax.grid(True, ls="--", alpha=0.3)

    ax_gt.set_xlabel("X [m]")
    ax_gt.set_ylabel("Y [m]")
    ax_pred.set_xlabel("X [m]")

    def update(frame: int):
        t_val = time_grid[frame].item()

        # Update Ground Truth
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
        ax_gt.grid(True, ls="--", alpha=0.3)

        # Update Prediction
        pos_pred = nodes_ref + u_pred_traj[frame]
        ax_pred.clear()
        ax_pred.set_title(f"MeshGraphNet [{method.upper()}] ($t = {t_val:.3f}$ s)")
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
        ax_pred.grid(True, ls="--", alpha=0.3)

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=time_steps, interval=1000 // fps, blit=False
    )
    save_path = output_dir / output_filename
    anim.save(save_path, writer="ffmpeg", fps=fps)
    plt.close()
    logger.info(f"Animation saved to: {save_path}")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    out_dir = Path(cfg.get("output_dir", "./output"))

    if cfg.get("bulk", False):
        model_paths = [
            p
            for p in out_dir.glob("*.pt")
            if not any(part in str(p) for part in ["checkpoint", "2D", "PINN"])
        ]
        if model_paths == []:
            model_paths = [
                p
                for p in out_dir.parent.rglob("*.pt")
                if not any(part in str(p) for part in ["checkpoint", "2D", "PINN"])
            ]

        for model_file in model_paths:
            try:
                print("=" * 10)
                print(model_file.stem)
                animate_deformation(model_path=model_file)
            except Exception as e:
                print("-" * 10)
                print(e)
    else:
        Tk().withdraw()
        selected = askopenfilename(initialdir=out_dir)
        if selected:
            animate_deformation(Path(selected))
