import time
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import torch

from src.geometry import create_mesh
from src.models import build_model
from src.utils import CSVLogger, load_config


def benchmark_inference(pinn, nodes, time_grid, device="cpu", n_warmup=20, n_runs=100):
    pinn.eval()
    num_nodes = nodes.shape[0]
    t_val = time_grid[0].item()
    t_in = torch.full((num_nodes, 1), t_val, device=device)
    X_in = nodes.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = pinn(t=t_in, X=X_in)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = pinn(t=t_in, X=X_in)
    elapsed = time.perf_counter() - start_time

    latency_per_step_ms = (elapsed / n_runs) * 1000.0

    return latency_per_step_ms


def pinn_loader(model_path):
    # Mesh
    x_range = cfg["domain"]["x_range"]
    y_range = cfg["domain"]["y_range"]
    nx, ny = int(cfg["domain"]["nx"]), int(cfg["domain"]["ny"])

    mesh = create_mesh(
        width=float(x_range[-1] - x_range[0]),
        height=float(y_range[-1] - y_range[0]),
        nx=nx,
        ny=ny,
        device=device,
    )
    nodes = mesh.nodes

    # Data
    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / cfg["data"]["dataset_name"]

    dataset = torch.load(dataset_path, map_location=device, weights_only=False)

    time_grid = dataset["time"].to(device)

    # Model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("config", cfg)
    pinn = build_model(saved_cfg, device=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    pinn.load_state_dict(state_dict)

    return pinn, nodes, time_grid


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    out_dir = Path(cfg["output_dir"])
    device = cfg["training"]["device"]
    latency_logger = CSVLogger(
        out_dir / f"time_results_{cfg['model']['type']}.csv",
        fieldnames=[
            "model_name",
            "time_latency_[ms/step]",
        ],
    )
    if cfg.get("bulk", False):
        pt_files = [p for p in out_dir.glob("*.pt") if "checkpoints" not in str(p)]
        for model_file in pt_files:
            try:
                pinn, nodes, time_grid = pinn_loader(model_file)
                latency = benchmark_inference(pinn, nodes, time_grid)
                result = {
                    "model_name": model_file.stem,
                    "time_latency_[ms/step]": latency,
                }
                latency_logger.log(result)
            except Exception as e:  # noqa: BLE001
                print(f"Error evaluating {model_file.stem}: {e}")
    else:
        Tk().withdraw()
        selected_model = askopenfilename(initialdir=out_dir)
        if selected_model:
            pinn, nodes, time_grid = pinn_loader(selected_model)
            latency = benchmark_inference(pinn, nodes, time_grid)
            result = {
                "model_name": Path(selected_model).stem,
                "time_latency_[ms/step]": latency,
            }
            latency_logger.log(result)
