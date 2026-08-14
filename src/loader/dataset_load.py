from pathlib import Path

import torch

from .elastic import elastic_generator
from .harmonics import harmonic_generator  # noqa: F401


def data_norm(data: torch.Tensor) -> tuple[torch.Tensor, float]:

    scale = data.max() - data.min()
    data = (data - data.min()) / scale

    return data, scale.detach().item()


def init_n_bound_data(
    cfg: dict, device: str | torch.device
) -> tuple[torch.Tensor, ...]:
    data_path = Path(cfg["data"]["data_dir"]) / cfg["data"]["dataset_name"]

    if not data_path.exists():
        elastic_generator(cfg)

    data = torch.load(data_path, map_location=device)

    return tuple(value.unsqueeze(-1) for value in data.values())


def test_data(
    cfg: dict, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
    x_min, x_max, x_points = cfg["testing"]["domain"]
    x = torch.linspace(x_min, x_max, x_points, device=device).unsqueeze(-1)
    A = float(cfg["testing"]["parameters"]["area"])
    E = float(cfg["testing"]["parameters"]["young"])
    F = float(cfg["testing"]["parameters"]["ext_force"])
    f0 = float(cfg["testing"]["parameters"]["int_force"])

    u_exact = (1.0 / (A * E)) * ((F + f0 * x_max) * x - (f0 / 2.0) * (x**2))
    ε_exact = (1.0 / (A * E)) * ((F + f0 * x_max) - f0 * x)

    return x, u_exact, ε_exact, F, A, E, f0
