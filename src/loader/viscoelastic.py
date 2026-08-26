from pathlib import Path

import numpy as np
import torch


def time_points(cfg: dict, save: bool = True) -> torch.Tensor:
    t0, T, t_step = cfg["synt"]["domain"]
    t = torch.linspace(t0, T, int(t_step)).view(-1, 1)

    if save:
        data_dir = Path(cfg["synt"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "time_points.pt"
        torch.save({"t": t}, file_path)

    return t


def visco_elastic_ode(
    t: float,
    E_vec: np.ndarray,
    Q: np.ndarray,
    B: np.ndarray,
    k: float,
    c: float,
    y_vec: np.ndarray,
) -> np.ndarray:
    s = float(E_vec.T @ Q @ E_vec)
    exp_s = c * np.exp(s)

    BE = B @ E_vec
    grad_psi = exp_s * BE

    BE_outer = np.outer(BE, BE)
    H = exp_s * (B + BE_outer)

    H_inv = np.linalg.inv(H)
    H_inv2 = H_inv @ H_inv

    E_t = -k * (H_inv2 @ (grad_psi - y_vec))
    return E_t
