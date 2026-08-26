from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .viscoelastic import time_points, visco_elastic_ode


def load_train_data(cfg: dict, device: str | torch.device) -> tuple[torch.Tensor, ...]:
    data_path = Path(cfg["data"]["data_dir"]) / cfg["data"]["dataset_name"]

    if not data_path.exists():
        time_points(cfg)

    data = torch.load(data_path, map_location=device)
    return tuple(value for value in data.values())


def generate_test_data(cfg: dict, save_path: Path) -> dict[str, torch.Tensor]:
    k = float(cfg["physics"]["relaxation_modulus"])
    c = float(cfg["physics"]["stress_scaling_factor"])
    c1, c2, c3 = [float(v) for v in cfg["physics"]["c_constants"]]
    y_vec = np.array(cfg["physics"]["stress"], dtype=np.float64)
    e0 = cfg["training"]["parameters"].get("intial_strain", [0.0, 0.0])

    t0, T, t_step = cfg["synt"]["domain"]
    t_eval_np = np.linspace(t0, T, int(t_step))

    Q = np.array([[c1, 0.5 * c3], [0.5 * c3, c2]], dtype=np.float64)
    B = 2.0 * Q

    sol = solve_ivp(
        visco_elastic_ode,
        (t0, T),
        list(e0),
        args=(Q, B, k, c, y_vec),
        t_eval=t_eval_np,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )

    t_tensor = torch.tensor(t_eval_np, dtype=torch.float32).view(-1, 1)
    E_exact_tensor = torch.tensor(sol.y.T, dtype=torch.float32)  # Shape: (N, 2)

    data = {"t": t_tensor, "E_exact": E_exact_tensor}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)
    return data


def test_data(
    cfg: dict, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    test_file = Path(cfg["data"]["data_dir"]) / "ground_truth_2D.pt"

    if not test_file.exists():
        data = generate_test_data(cfg, test_file)
    else:
        data = torch.load(test_file, map_location=device)

    t_test = data["t"].to(device)
    E_exact = data["E_exact"].to(device)
    return t_test, E_exact
