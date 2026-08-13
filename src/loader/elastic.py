from pathlib import Path

import torch


def elastic_generator_basic(
    cfg: dict, save: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0, L, x_step = cfg["synt"]["period"]
    x = torch.linspace(x0, L, x_step)

    A = cfg["synt"]["parameters"]["area"]
    E = cfg["synt"]["parameters"]["young"]
    F = cfg["synt"]["parameters"]["ext_force"]
    f0 = cfg["synt"]["parameters"]["int_force"]

    u = 1 / (A * E) * ((F + f0 * L) * x - f0 / 2 * x**2)
    ε = 1 / (A * E) * ((F + f0 * L) - f0 * x)  # ε = du/dx

    if save:
        data_dir = Path(cfg["synt"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "elastic_1d_data.pt"
        torch.save({"x": x, "u": u, "ε": ε}, file_path)

    return x, u, ε

def elastic_generator(
    cfg: dict, save: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0, L, x_step = cfg["synt"]["period"]
    x = torch.linspace(x0, L, x_step)

    A = cfg["synt"]["parameters"]["area"]
    E = cfg["synt"]["parameters"]["young"]
    F = cfg["synt"]["parameters"]["ext_force"]
    f0 = cfg["synt"]["parameters"]["int_force"]

    u = 1 / (A * E) * ((F + f0 * L) * x - f0 / 2 * x**2)
    ε = 1 / (A * E) * ((F + f0 * L) - f0 * x)  # ε = du/dx

    if save:
        data_dir = Path(cfg["synt"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "elastic_1d_data.pt"
        torch.save({"x": x, "u": u, "ε": ε}, file_path)

    return x, u, ε
