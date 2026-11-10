from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..geometry import create_mesh_3D
from ..loader import generate_inertial_dataset
from ..utils import init_logging, load_config
from .animate_dataset import animate_ground_truth


def show_dataset(trac_type: str, dtype: torch.dtype = torch.float32):
    init_logging()
    cfg = load_config()
    device = torch.device("cpu")
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset loading / generation
    data_dir = Path(cfg["data"]["data_dir"])
    data_name = f"F_{trac_type}_3D.pt"  # str(cfg["data"]["dataset_name"])
    dataset_path = data_dir / data_name

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Generating now...")
        dataset = generate_inertial_dataset(cfg, device=device, trac_type=trac_type)
    else:
        print(f"Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

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

    time_axis = dataset["time"].to(dtype=dtype)
    displacement = dataset["u"].to(dtype=dtype)
    volume_weights = torch.zeros(mesh.n_nodes, dtype=dtype, device=device)
    volume_weights.index_add_(
        0,
        mesh.elements.reshape(-1),
        (mesh.elements_volume.to(dtype=dtype) / 4).repeat_interleave(4),
    )
    average_displacement = (displacement * volume_weights[None, :, None]).sum(
        dim=1
    ) / volume_weights.sum()

    left_nodes = mesh.boundary_nodes["left"]
    left_node_weights = torch.zeros(mesh.n_nodes, dtype=dtype, device=device)
    left_node_weights.index_add_(
        0,
        mesh.boundary_faces["left"].reshape(-1),
        (mesh.boundary_areas["left"].to(dtype=dtype) / 3).repeat_interleave(3),
    )
    average_clamp_displacement = (
        displacement[:, left_nodes, :] * left_node_weights[left_nodes][None, :, None]
    ).sum(dim=1) / left_node_weights.sum()

    traction = dataset["trac_ext"].to(dtype=dtype)
    if traction.ndim == 2:
        traction = traction[:, None, :].expand(
            -1, len(mesh.boundary_nodes["right"]), -1
        )
    right_nodes = mesh.boundary_nodes["right"]
    right_node_weights = torch.zeros(mesh.n_nodes, dtype=dtype, device=device)
    right_node_weights.index_add_(
        0,
        mesh.boundary_faces["right"].reshape(-1),
        (mesh.boundary_areas["right"].to(dtype=dtype) / 3).repeat_interleave(3),
    )
    right_node_weights = right_node_weights[right_nodes]
    weighted_traction = traction * right_node_weights[None, :, None]
    average_traction = weighted_traction.sum(dim=1) / right_node_weights.sum()
    resultant_force = weighted_traction.sum(dim=1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    component_labels = (r"$x$", r"$y$", r"$z$")

    def plot_components(ax, values: torch.Tensor, title: str, ylabel: str):
        values = values.detach().cpu()
        for component, label in enumerate(component_labels):
            ax.plot(time_axis.cpu(), values[:, component], label=label)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.legend(title="Component")
        ax.grid(True, which="both", ls="--", alpha=0.3)

    axs = axes.flat
    plot_components(
        axs[0], average_displacement, "Average Displacement", "Displacement [m]"
    )
    axs[1].plot(time_axis, dataset["psi"].to(dtype=dtype), label="Internal energy")
    axs[1].set_title("Internal Energy")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Energy [J]")
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[1].legend()
    axs[1].grid(True, which="both", ls="--", alpha=0.3)
    plot_components(
        axs[2],
        average_clamp_displacement,
        "Average Clamp Displacement",
        "Displacement [m]",
    )
    plot_components(axs[3], average_traction, "Average Right Traction", "Traction [Pa]")
    plot_components(axs[4], resultant_force, "Resultant Load", "Force [N]")
    axs[5].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / f"{Path(data_name).stem}.png")
    plt.close(fig)
    print("Dataset plot saved on", output_dir / f"{Path(data_name).stem}.png")


if __name__ == "__main__":
    trac_types = ["const", "ramp", "half_sin", "sin", "step", "impulse"]
    for trac_type in trac_types:
        try:
            show_dataset(trac_type=trac_type)
            animate_ground_truth(data_name=f"F_{trac_type}_3D.pt")
        except:
            print(trac_type, " didn't converge")
