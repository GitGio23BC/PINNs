import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.geometry import create_graph, create_mesh_3D
from src.loader import MGNData
from src.models import MeshGraphNet
from src.utils import load_config


def next_available_path(path: Path) -> Path:
    candidate = path
    version = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{version}{path.suffix}")
        version += 1
    return candidate


def rms_free(
    values: torch.Tensor,
    free_dofs: torch.Tensor,
) -> torch.Tensor:
    values_flat = values.reshape(values.shape[0], -1)
    return values_flat[:, free_dofs].square().mean(dim=1).sqrt()


def plot_global_rms(
    ax,
    times,
    pred,
    ref_fd,
    ref_saved,
    *,
    title: str,
    ylabel: str,
) -> None:
    ax.plot(times, pred, label="Predicted", color="tab:red")
    ax.plot(times, ref_fd, label="Reference finite difference", linestyle="--")
    ax.plot(times, ref_saved, label="Reference saved", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()


def plot_node_components(
    ax,
    times,
    pred: torch.Tensor,
    ref_fd: torch.Tensor,
    ref_saved: torch.Tensor,
    node: int,
    *,
    title: str,
    ylabel: str,
) -> None:
    component_names = ("x", "y", "z")
    component_colors = ("tab:red", "tab:green", "tab:blue")
    methods = (
        (pred, "Predicted", "-"),
        (ref_fd, "Reference FD", "--"),
        (ref_saved, "Reference saved", ":"),
    )

    for component, (component_name, color) in enumerate(
        zip(component_names, component_colors)
    ):
        for values, method_name, linestyle in methods:
            ax.plot(
                times,
                values[:, node, component],
                color=color,
                linestyle=linestyle,
                label=f"{method_name} {component_name}",
            )

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7, ncol=3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot predicted and reference velocity/acceleration."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint path; defaults to the model configured in config.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    device = torch.device(cfg["training"].get("device", "cuda"))
    dtype = torch.float32

    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    z_min, z_max = cfg["domain"]["z_range"]

    mesh = create_mesh_3D(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        depth=float(z_max - z_min),
        nx=int(cfg["domain"]["nx"]),
        ny=int(cfg["domain"]["ny"]),
        nz=int(cfg["domain"]["nz"]),
        device=device,
        dtype=dtype,
    )

    data_path = Path(cfg["data"]["data_dir"]) / cfg["data"]["dataset_name"]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = torch.load(data_path, map_location=device, weights_only=False)
    data_loader = MGNData(
        dataset=dataset,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )
    data = data_loader.dataset

    required = ("u", "u_t", "u_tt", "trac_ext")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset missing required fields: {missing}")

    model_cfg = cfg["model"]
    default_model = (
        Path(cfg["output_dir"])
        / str(model_cfg["model_name"])
        / f"{model_cfg['model_name']}.pt"
    )
    model_file = args.model or default_model
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state_dict)
    model.eval()

    directions = ("left", "right", "bottom", "top", "back", "front")
    node_type = torch.zeros(
        (mesh.n_nodes, len(directions)),
        device=device,
        dtype=dtype,
    )
    for index, direction in enumerate(directions):
        node_type[mesh.get_boundary_nodes(direction), index] = 1.0

    dt = data_loader.dt
    trac_scale = float(cfg["physics"]["sigma_max"])

    u_prev = torch.zeros((mesh.n_nodes, 3), device=device, dtype=dtype)
    u_prev2 = torch.zeros_like(u_prev)
    velocity_prev = torch.zeros_like(u_prev)
    acceleration_prev = torch.zeros_like(u_prev)
    world_pos = mesh.nodes.clone()

    u_ref_prev = torch.zeros_like(u_prev)
    u_ref_prev2 = torch.zeros_like(u_prev)

    predicted_ut = []
    predicted_utt = []
    reference_fd_ut = []
    reference_fd_utt = []
    reference_saved_ut = []
    reference_saved_utt = []
    rows = []

    for step in range(data_loader.num_steps):
        batch = data_loader.get_batch(step)
        trac_feat = torch.zeros(
            (mesh.n_nodes, 3),
            device=device,
            dtype=dtype,
        )
        trac_feat[mesh.get_boundary_nodes("right")] = (
            batch.trac_ext / trac_scale
        )

        graph = create_graph(
            mesh=mesh,
            world_pos=world_pos,
            node_type=node_type,
            velocity=velocity_prev,
            acceleration=acceleration_prev,
            boundary_traction=trac_feat,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            u_pred = model(graph).to(dtype=dtype)

        u_t_pred = (u_pred - u_prev) / dt
        u_tt_pred = (u_pred - 2.0 * u_prev + u_prev2) / (dt**2)

        u_ref = data["u"][step].detach()
        u_t_ref_saved = data["u_t"][step].detach()
        u_tt_ref_saved = data["u_tt"][step].detach()

        # Reference derivatives using the same finite differences as the rollout.
        u_t_ref_fd = (u_ref - u_ref_prev) / dt
        u_tt_ref_fd = (
            u_ref - 2.0 * u_ref_prev + u_ref_prev2
        ) / (dt**2)

        predicted_ut.append(u_t_pred.detach().cpu())
        predicted_utt.append(u_tt_pred.detach().cpu())
        reference_fd_ut.append(u_t_ref_fd.detach().cpu())
        reference_fd_utt.append(u_tt_ref_fd.detach().cpu())
        reference_saved_ut.append(u_t_ref_saved.detach().cpu())
        reference_saved_utt.append(u_tt_ref_saved.detach().cpu())

        u_prev2 = u_prev.detach().clone()
        u_prev = u_pred.detach().clone()
        velocity_prev = ((u_prev - u_prev2) / dt).detach()
        acceleration_prev = u_tt_pred.detach().clone()
        world_pos = (mesh.nodes + u_prev).detach()

        u_ref_prev2 = u_ref_prev.detach().clone()
        u_ref_prev = u_ref.detach().clone()

    predicted_ut = torch.stack(predicted_ut)
    predicted_utt = torch.stack(predicted_utt)
    reference_fd_ut = torch.stack(reference_fd_ut)
    reference_fd_utt = torch.stack(reference_fd_utt)
    reference_saved_ut = torch.stack(reference_saved_ut)
    reference_saved_utt = torch.stack(reference_saved_utt)

    free_dofs_cpu = mesh.free_dofs.detach().cpu()
    times = data_loader.time.detach().cpu().numpy()

    # Find representative free nodes with the largest trajectory RMS error.
    free_dof_mask = torch.ones(mesh.n_dofs, dtype=torch.bool, device=device)
    free_dof_mask[mesh.fixed_dofs] = False
    free_node_mask = free_dof_mask.reshape(mesh.n_nodes, 3).any(dim=1)
    free_nodes = torch.nonzero(free_node_mask, as_tuple=False).squeeze(1).cpu()

    velocity_node_error = (
        (predicted_ut - reference_fd_ut)
        .square()
        .sum(dim=-1)
        .mean(dim=0)
        .sqrt()
    )
    acceleration_node_error = (
        (predicted_utt - reference_fd_utt)
        .square()
        .sum(dim=-1)
        .mean(dim=0)
        .sqrt()
    )

    velocity_node = int(
        free_nodes[
            torch.argmax(velocity_node_error[free_nodes])
        ].item()
    )
    acceleration_node = int(
        free_nodes[
            torch.argmax(acceleration_node_error[free_nodes])
        ].item()
    )

    ut_pred_rms = rms_free(predicted_ut, free_dofs_cpu)
    ut_ref_fd_rms = rms_free(reference_fd_ut, free_dofs_cpu)
    ut_ref_saved_rms = rms_free(reference_saved_ut, free_dofs_cpu)

    utt_pred_rms = rms_free(predicted_utt, free_dofs_cpu)
    utt_ref_fd_rms = rms_free(reference_fd_utt, free_dofs_cpu)
    utt_ref_saved_rms = rms_free(reference_saved_utt, free_dofs_cpu)

    for step, time_s in enumerate(times):
        row = {
            "step": step,
            "time_s": float(time_s),
            "u_t_pred_rms_free_m_per_s": float(ut_pred_rms[step]),
            "u_t_ref_fd_rms_free_m_per_s": float(ut_ref_fd_rms[step]),
            "u_t_ref_saved_rms_free_m_per_s": float(ut_ref_saved_rms[step]),
            "u_tt_pred_rms_free_m_per_s2": float(utt_pred_rms[step]),
            "u_tt_ref_fd_rms_free_m_per_s2": float(utt_ref_fd_rms[step]),
            "u_tt_ref_saved_rms_free_m_per_s2": float(
                utt_ref_saved_rms[step]
            ),
            "velocity_node": velocity_node,
            "acceleration_node": acceleration_node,
        }

        for component, axis in enumerate(("x", "y", "z")):
            row[f"velocity_node_pred_{axis}"] = float(
                predicted_ut[step, velocity_node, component]
            )
            row[f"velocity_node_ref_fd_{axis}"] = float(
                reference_fd_ut[step, velocity_node, component]
            )
            row[f"velocity_node_ref_saved_{axis}"] = float(
                reference_saved_ut[step, velocity_node, component]
            )

            row[f"acceleration_node_pred_{axis}"] = float(
                predicted_utt[step, acceleration_node, component]
            )
            row[f"acceleration_node_ref_fd_{axis}"] = float(
                reference_fd_utt[step, acceleration_node, component]
            )
            row[f"acceleration_node_ref_saved_{axis}"] = float(
                reference_saved_utt[step, acceleration_node, component]
            )

        rows.append(row)

    output_dir = Path(cfg.get("output_dir", "./output")) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = next_available_path(
        output_dir / f"{model_file.stem}_ut_utt_timeseries.csv"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    velocity_coords = mesh.nodes[velocity_node].detach().cpu().tolist()
    acceleration_coords = (
        mesh.nodes[acceleration_node].detach().cpu().tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    plot_global_rms(
        axes[0, 0],
        times,
        ut_pred_rms.numpy(),
        ut_ref_fd_rms.numpy(),
        ut_ref_saved_rms.numpy(),
        title="Velocity RMS over free DOFs",
        ylabel="RMS velocity [m/s]",
    )
    plot_global_rms(
        axes[0, 1],
        times,
        utt_pred_rms.numpy(),
        utt_ref_fd_rms.numpy(),
        utt_ref_saved_rms.numpy(),
        title="Acceleration RMS over free DOFs",
        ylabel="RMS acceleration [m/s²]",
    )

    plot_node_components(
        axes[1, 0],
        times,
        predicted_ut,
        reference_fd_ut,
        reference_saved_ut,
        velocity_node,
        title=(
            f"Velocity components at node {velocity_node}; "
            f"coords={velocity_coords}"
        ),
        ylabel="Velocity [m/s]",
    )
    plot_node_components(
        axes[1, 1],
        times,
        predicted_utt,
        reference_fd_utt,
        reference_saved_utt,
        acceleration_node,
        title=(
            f"Acceleration components at node {acceleration_node}; "
            f"coords={acceleration_coords}"
        ),
        ylabel="Acceleration [m/s²]",
    )

    plot_path = next_available_path(
        output_dir / f"{model_file.stem}_ut_utt_timeseries.png"
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved time-series CSV to: {csv_path}")
    print(f"Saved time-series plot to: {plot_path}")
    print(
        f"Velocity node: {velocity_node}, coordinates={velocity_coords}"
    )
    print(
        f"Acceleration node: {acceleration_node}, "
        f"coordinates={acceleration_coords}"
    )


if __name__ == "__main__":
    main()
