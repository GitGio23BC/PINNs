from pathlib import Path

import torch


def time_points(
    cfg: dict, save: bool = True
) -> torch.Tensor:
    t0, T, t_step = cfg["synt"]["domain"]
    t = torch.linspace(t0, T, int(t_step)).view(-1, 1)

    if save:
        data_dir = Path(cfg["synt"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "time_points.pt"
        torch.save({"t": t}, file_path)

    return t