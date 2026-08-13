from pathlib import Path

import torch


def harmonic_generator(
    cfg: dict, save: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    t0, tf, t_step = cfg["synt"]["period"]
    t = torch.linspace(t0, tf, t_step)

    A = cfg["synt"]["amplitude"]
    phi = cfg["synt"]["phase"]
    omega = cfg["synt"]["damping_frequency"]
    zeta = cfg["synt"]["damping_coefficient"]

    if isinstance(A, (list, tuple)):
        ux = A[0] * torch.exp(-zeta[0] * t) * torch.cos(omega[0] * t + phi[0])
        uy = A[1] * torch.exp(-zeta[1] * t) * torch.cos(omega[1] * t + phi[1])
        u = torch.stack([ux, uy], dim=-1)
    else:
        u = A * torch.exp(-zeta * t) * torch.cos(omega * t + phi)

    if save:
        data_dir = Path(cfg["synt"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "harmonic_data.pt"
        torch.save({"t": t, "u": u}, file_path)

    return t, u
