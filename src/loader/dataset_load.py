from pathlib import Path

import torch

from .viscoelastic import time_points


def load_train_data(
    cfg: dict, device: str | torch.device
) -> tuple[torch.Tensor, ...]:
    data_path = Path(cfg["data"]["data_dir"]) / cfg["data"]["dataset_name"]

    if not data_path.exists():
        time_points(cfg)

    data = torch.load(data_path, map_location=device)

    return tuple(value for value in data.values())


def test_data(
    cfg: dict, device: str | torch.device
) -> None:
    # TODO
    pass
