from pathlib import Path

import torch

from .elastic import elastic_generator
from .harmonics import harmonic_generator  # noqa: F401


def data_norm(data: torch.Tensor) -> tuple[torch.Tensor, float]:

    scale = data.max() - data.min()
    data = (data - data.min()) / scale

    return data, scale.detach().item()


def init_n_bound_data(cfg: dict, device: str) -> tuple[torch.Tensor, ...]:
    data_path = Path(cfg["data"]["data_dir"]) / cfg["data"]["dataset_name"]

    if not data_path.exists():
        elastic_generator(cfg)

    data = torch.load(data_path, map_location=device)

    return tuple(value.unsqueeze(-1) for value in data.values())
