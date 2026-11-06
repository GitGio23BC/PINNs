from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..geometry import create_mesh
from ..loader import MGNData, generate_inertial_dataset
from ..utils import init_logging, load_config
from .animate_dataset import animate_ground_truth


def show_dataset(dtype: torch.dtype = torch.float32):
    init_logging()
    cfg = load_config()
    device = torch.device("cpu")
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset loading / generation
    data_dir = Path(cfg["data"]["data_dir"])
    data_name = cfg["data"]["dataset_name"]
    dataset_path = data_dir / data_name

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Generating now...")
        dataset = generate_inertial_dataset(cfg, device=device)
    else:
        print(f"Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    # Mesh creation
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
    data_loader = MGNData(
        dataset=dataset,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )

    time_axis = data_loader.time
    data = data_loader.get_trajectory()

    fig, axes = plt.subplots(2, 3, figsize=(20,10))
    df = {
        "Average X Displacement": (
            "u_x [m/m]",
            data.u_exact[:, :, 0].mean(dim=1),
        ),
        "Internal Energy": ("ψ [J/m³]", data.psi_exact),
        "Average Clamp X Displacement": (
            "u_x [m/m]",
            data.u_exact[:, mesh.left_nodes, 0].mean(dim=1),
        ),
        "Average X Traction": ("trac_x [Pa]", data.trac_ext[:, :, 0].mean(dim=1)),
        "Average X Force": ("F_ext,x [N]", data.F_ext[:, :, 0].mean(dim=1)),
    }
    axs = axes.flat
    for idx, (key, name_data) in enumerate(df.items()):
        axs[idx].plot(
            time_axis,
            name_data[1],
            linestyle="-",
            alpha=1.0,
            label=key,
        )
        axs[idx].set_title(key)
        axs[idx].set_xlabel("Time [s]")
        axs[idx].set_ylabel(name_data[0])
        axs[idx].legend()
        axs[idx].grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"{Path(data_name).stem}.png")
    plt.close(fig)
    print("Dataset plot saved on ", {output_dir / f"{Path(data_name).stem}.png"})

if __name__ == "__main__":
    show_dataset()
    animate_ground_truth()