from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ..geometry import create_mesh
from ..physics import HUGO, HolzapfelEnergy_2D


def generate_ground_truth(
    cfg: dict,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:

    data_dir = Path(cfg["data"]["data_dir"])
    dataset_name = cfg["data"]["dataset_name"]
    save_file = data_dir / dataset_name
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Domain and Mesh
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    nx = int(cfg["domain"].get("nx", 10))
    ny = int(cfg["domain"].get("ny", 10))
    width = float(x_max - x_min)
    height = float(y_max - y_min)

    mesh = create_mesh(
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        device="cpu",
    )
    nodes = mesh.nodes
    num_nodes = mesh.n_nodes
    tip_nodes_idx = mesh.right_nodes

    # Temporal Grid
    t0, t_final = cfg["domain"]["t_range"]
    n_steps = int(cfg["domain"].get("n_steps", 100))
    t_eval = np.linspace(float(t0), float(t_final), n_steps)

    # Physics Constants
    phys_cfg = cfg["physics"]
    c = float(phys_cfg["c"])
    c1 = float(phys_cfg["c1"])
    c2 = float(phys_cfg["c2"])
    c3 = float(phys_cfg["c3"])
    k_relax = float(phys_cfg["k_relax"])
    sigma_max = float(phys_cfg.get("sigma_max", 1.0))

    psi = HolzapfelEnergy_2D(c=c, c1=c1, c2=c2, c3=c3, device="cpu")
    visco_model = HUGO(psi=psi, k_relax=k_relax)

    def applied_stress_voigt(t: float) -> np.ndarray:
        s11 = sigma_max * 0.5 * (1.0 - np.cos(2.0 * np.pi * t / float(t_final)))
        return np.array([s11, 0.0, 0.0], dtype=np.float32)

    def rhs_haslach(t: float, E_flat: np.ndarray) -> np.ndarray:
        E_t = torch.from_numpy(E_flat.astype(np.float32)).unsqueeze(0)
        S_t = torch.from_numpy(applied_stress_voigt(t)).unsqueeze(0)

        with torch.no_grad():
            E_dot = visco_model.haslach_equation(E_t, S_t)

        return E_dot.squeeze(0).numpy().astype(np.float64)

    E0 = np.zeros(3, dtype=np.float64)
    sol = solve_ivp(
        fun=rhs_haslach,
        t_span=(float(t0), float(t_final)),
        y0=E0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-7,
        atol=1e-9,
    )

    E_single_node = torch.from_numpy(sol.y.T).float()
    E_trajectory = E_single_node.unsqueeze(1).expand(-1, num_nodes, -1).clone()

    lambda_1 = torch.sqrt(torch.clamp(2.0 * E_trajectory[..., 0] + 1.0, min=1e-6))
    lambda_2 = torch.sqrt(torch.clamp(2.0 * E_trajectory[..., 1] + 1.0, min=1e-6))

    u_trajectory = torch.zeros((n_steps, num_nodes, 2), dtype=torch.float32)
    u_trajectory[..., 0] = (lambda_1 - 1.0) * nodes[:, 0].unsqueeze(0)
    u_trajectory[..., 1] = (lambda_2 - 1.0) * nodes[:, 1].unsqueeze(0)

    with torch.no_grad():
        E_flat = E_trajectory.reshape(-1, 3)
        S_flat = psi.grad(E_flat)
        S_trajectory = S_flat.reshape(n_steps, num_nodes, 3)

    s11_applied = torch.tensor(
        [applied_stress_voigt(t)[0] for t in t_eval], dtype=torch.float32
    )
    trac_ext_trajectory = torch.zeros((n_steps, num_nodes, 2), dtype=torch.float32)
    trac_ext_trajectory[:, tip_nodes_idx, 0] = s11_applied.unsqueeze(1)

    dataset = {
        "time": torch.from_numpy(t_eval).float().to(device),
        "nodes": mesh.nodes.to(device),
        "elements": mesh.elements.to(device),
        "edges": mesh.edges.to(device),
        "boundary_nodes": mesh.boundary_nodes.to(device),
        "left_nodes": mesh.left_nodes.to(device),
        "right_nodes": mesh.right_nodes.to(device),
        "bottom_nodes": mesh.bottom_nodes.to(device),
        "top_nodes": mesh.top_nodes.to(device),
        "u": u_trajectory.to(device),
        "E": E_trajectory.to(device),
        "S": S_trajectory.to(device),
        "trac_ext": trac_ext_trajectory.to(device),
    }

    torch.save(dataset, save_file)
    return dataset
